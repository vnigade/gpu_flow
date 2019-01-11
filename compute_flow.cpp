//************************************************************************
// compute_flow.cpp
// Computes OpenCV GPU Brox et al. [1] and Zach et al. [2] TVL1 Optical Flow
// Dependencies: OpenCV and Qt5 for iterating (sub)directories
// Author: Christoph Feichtenhofer
// Institution: Graz University of Technology
// Email: feichtenhofer@tugraz
// Date: Nov. 2015
// [1] T. Brox, A. Bruhn, N. Papenberg, J. Weickert. High accuracy optical flow
// estimation based on a theory for warping. ECCV 2004.
// [2] C. Zach, T. Pock, H. Bischof: A duality based approach for realtime TV-L
// 1 optical flow. DAGM 2007.
//************************************************************************
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cstdlib>
#include <string>
#include <vector>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <time.h>
#include <sstream>
#include <dirent.h>

#include <QDirIterator>
#include <QFileInfo>
#include <QString>

#include "opencv2/core.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"

#define FRAME_FILE_PREFIX "frame"
#define FRAME_FILE_EXT ".jpg"
#define WRITEOUT_IMGS 1

// Debug LOG macros
#if defined(DEBUG)
#define LOG(msg) std::cout << msg << std::endl
#else
#define LOG(msg)
#endif

using namespace std;
using namespace cv;

float MIN_SZ = 256;
float OUT_SZ = 256;

bool clipFlow = true;  // clips flow to [-20 20]
bool resize_img = true;

bool createOutDirs = true;
bool rgb = false;

// Global variables for BroxOpticalFlow
const float alpha_ = 0.197;
const float gamma_ = 50;
const float scale_factor_ = 0.8;
const int inner_iterations_ = 10;
const int outer_iterations_ = 77;
const int solver_iterations_ = 10;

// Global variables for Farneback
const int numLevels = 5;
const double pyrScale = 0.5;
const int winSize = 13;
const int numIters = 10;
const int polyN = 5;
const double polySigma = 1.1;
const int flags = 0;

static void convertFlowToImage(const Mat &flowIn, Mat &flowOut,
                               float lowerBound, float higherBound) {
#define CAST(v, L, H) \
  ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255 * ((v) - (L)) / ((H) - (L))))
  for (int i = 0; i < flowIn.rows; ++i) {
    for (int j = 0; j < flowIn.cols; ++j) {
      float x = flowIn.at<float>(i, j);
      flowOut.at<uchar>(i, j) = CAST(x, lowerBound, higherBound);
    }
  }
#undef CAST
}

int main(int argc, char *argv[]) {
  cv::cuda::GpuMat frame0GPU, frame1GPU, uGPU, vGPU;
  Mat frame_iter, frame0_rgb, frame1_rgb, frame0_grey, frame1_grey, rgb_out;
  Mat frame0_grey_32f, frame1_grey_32f, imgU, imgV;

  double t1 = 0.0, t2 = 0.0, tdflow = 0.0, t1fr = 0.0, t2fr = 0.0,
         tdframe = 0.0;
  double avg_tdframe = 0.0, avg_tdflow = 0.0;
  int gpuID = 0;
  int type = 1;
  int frameSkip = 1;
  std::string vid_path, out_path, out_path_jpeg;

  const char *keys =
      "{ h help usage ?|       | print help message }"
      "{ g gpuID       |  0    | use this gpu}"
      "{ f type        |  0    | use this flow method (0=Brox, 1=TV-L1, "
      "2=Farneback)}"
      "{ s skip        |  1    | frame skip}"
      "{ @vid_path     |  ./   | path input (where the videos are)}"
      "{ @out_path     |  ./   | path output}";

  CommandLineParser cmd(argc, argv, keys);
  if (cmd.has("h") || argc < 1) {
    cmd.printMessage();
    return 0;
  }

  gpuID = cmd.get<int>("gpuID");
  type = cmd.get<int>("type");
  frameSkip = cmd.get<int>("skip");
  vid_path = cmd.get<std::string>("@vid_path");
  out_path = cmd.get<std::string>("@out_path");
  out_path_jpeg = out_path + "/rgb/";
  cout << "GpuID: " << gpuID << "\n"
       << "FlowMethod: " << type << "\n"
       << "FrameSkip: " << frameSkip << "\n"
       << "VidPath: " << vid_path << "\n"
       << "OutPath: " << out_path << "\n"
       << "Jpegs: " << out_path_jpeg << endl;

#if defined(DEBUG)
  DIR *dirp;
  struct dirent *entry;
  int totalvideos = 0, vidcount = 0;
  // count number of video files
  dirp = opendir(vid_path.c_str());
  if (dirp != NULL) {
    while ((entry = readdir(dirp)) != NULL) {
      if (entry->d_type == DT_REG || entry->d_type == DT_LNK) {
        totalvideos++;
      }
    }
    closedir(dirp);
  } else {
    exit(EXIT_FAILURE);
  }
#endif

  std::cout << "Cuda enabled devices: " << cuda::getCudaEnabledDeviceCount() << std::endl;
  cv::cuda::setDevice(gpuID);
  Mat capture_frame, capture_image, prev_image, capture_gray, prev_gray,
      human_mask;
  cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

  cv::Ptr<cv::cuda::BroxOpticalFlow> brox = cuda::BroxOpticalFlow::create(
      alpha_, gamma_, scale_factor_, inner_iterations_, outer_iterations_,
      solver_iterations_);
  cv::Ptr<cv::cuda::OpticalFlowDual_TVL1> alg_tvl1 =
      cv::cuda::OpticalFlowDual_TVL1::create();
  cv::Ptr<cv::cuda::FarnebackOpticalFlow> d_farneback =
      cv::cuda::FarnebackOpticalFlow::create(numLevels, pyrScale, false,
                                             winSize, numIters, polyN,
                                             polySigma, flags);

  QString vpath = QString::fromStdString(vid_path);
  QDirIterator dirIt(vpath, QDirIterator::Subdirectories);
  std::string video, outfile_u, outfile_v, outfile_jpeg;

  // process all videos in the directory
  while (dirIt.hasNext()) {
    dirIt.next();
    QFileInfo fileInfo(dirIt.filePath());
    if (!(fileInfo.suffix() == "mp4") && !(fileInfo.suffix() == "avi")) {
      continue;
    }

    std::string path = fileInfo.absolutePath().toStdString();
    std::string fName = fileInfo.baseName().toStdString();

    QString out_folder_u = QString::fromStdString(out_path + "x/" + fName);
    QString out_folder_v = QString::fromStdString(out_path + "y/" + fName);
    if (QDir(out_folder_u).exists() || QDir(out_folder_v).exists()) {
      std::cout << "already exists: " << out_path << fName << std::endl;
      continue;
    }

    if (!(QDir().mkpath(out_folder_u) && QDir().mkpath(out_folder_v))) {
      std::cout << "cannot create: " << out_path << fName << std::endl;
      continue;
    }
    outfile_u = out_folder_u.toStdString();
    outfile_v = out_folder_v.toStdString();

    if (rgb) {
      outfile_jpeg = out_path_jpeg + fName;
      QString out_folder_jpeg = QString::fromStdString(outfile_jpeg);
      if (!QDir(out_folder_jpeg).exists()) {
        QDir().mkpath(out_folder_jpeg);
      }
    }

#if defined(DEBUG)
    LOG(fileInfo.absoluteFilePath().toStdString() << "    " << vidcount << "/"
                                                  << totalvideos);
    vidcount++;
#endif

    VideoCapture cap;
    try {
      cap.open(fileInfo.absoluteFilePath().toStdString());  // read video from
                                                            // the filesystem
    }
    catch (std::exception &e) {
      std::cout << e.what() << '\n';
    }

    if (cap.isOpened() == 0) {
      exit(EXIT_FAILURE);
    }

    int nframes = 0, width = 0, height = 0;
    float factor = 0, factor_out = 0;
    cap >> frame_iter;  // next frame.

    if (resize_img == true) {
      factor =
          std::max<float>(MIN_SZ / frame_iter.cols, MIN_SZ / frame_iter.rows);
      width = std::floor(frame_iter.cols * factor);
      width -= width % 2;
      height = std::floor(frame_iter.rows * factor);
      height -= height % 2;
      factor_out = std::max<float>(OUT_SZ / width, OUT_SZ / height);
    } else {
      width = frame_iter.cols;
      height = frame_iter.rows;
    }

    // Allocate memory for the images
    frame1_rgb = cv::Mat(Size(width, height), CV_8UC3);
    frame0_rgb = cv::Mat(Size(width, height), CV_8UC3);
    frame0_grey = cv::Mat(Size(width, height), CV_8UC1);
    frame1_grey = cv::Mat(Size(width, height), CV_8UC1);
    frame0_grey_32f = cv::Mat(Size(width, height), CV_32FC1);
    frame1_grey_32f = cv::Mat(Size(width, height), CV_32FC1);
    rgb_out =
        cv::Mat(Size(cvRound(width * factor_out), cvRound(height * factor_out)),
                CV_8UC3);

    // Iterate all frames and skip some if mentioned.
    while (frame_iter.empty() == false) {
      t1fr = cv::getTickCount();

      // Convert frames
      if (resize_img == true) {
        cv::resize(frame_iter, frame1_rgb, cv::Size(width, height), 0, 0,
                   INTER_CUBIC);
      } else {
        frame_iter.copyTo(frame1_rgb);
      }

      // Save RGB frame
      // OpenCV seems to convert the image from BGR to RGB while writing
      if (rgb) {
        std::ostringstream strStream;
        strStream << outfile_jpeg << "/" << FRAME_FILE_PREFIX << (nframes + 1)
                  << FRAME_FILE_EXT;
        if (resize_img == true) {
          cv::resize(frame1_rgb, rgb_out, cv::Size(width, height), 0, 0,
                     INTER_CUBIC);
          imwrite(strStream.str(), rgb_out);
        } else {
          imwrite(strStream.str(), frame1_rgb);
        }
      }

      // calculate optical flow
      if (nframes >= 1) {
        cv::cvtColor(frame1_rgb, frame1_grey, CV_BGR2GRAY);
        frame1_grey.convertTo(frame1_grey_32f, CV_32FC1, 1.0 / 255.0, 0);

        t1 = cv::getTickCount();
        cv::cuda::GpuMat flow, flows[2];
        switch (type) {
          case 0:
            frame1GPU.upload(frame1_grey_32f);
            frame0GPU.upload(frame0_grey_32f);
            brox->calc(frame0GPU, frame1GPU, flow);
            break;

          case 1:
            frame1GPU.upload(frame1_grey);
            frame0GPU.upload(frame0_grey);
            alg_tvl1->calc(frame0GPU, frame1GPU, flow);
            break;

          case 2:
            frame1GPU.upload(frame1_grey);
            frame0GPU.upload(frame0_grey);
            d_farneback->calc(frame0GPU, frame1GPU, flow);
        }
        cv::cuda::split(flow, flows);
        uGPU = flows[0];
        vGPU = flows[1];

        uGPU.download(imgU);
        vGPU.download(imgV);

        t2 = cv::getTickCount();
        tdflow = 1000.0 * (t2 - t1) / cv::getTickFrequency();
        avg_tdflow = avg_tdflow + (tdflow - avg_tdflow) / nframes;
      }

      if (WRITEOUT_IMGS == true && nframes >= 1) {
        if (resize_img == true) {  // resize optical flow
          cv::resize(imgU, imgU, cv::Size(width, height), 0, 0, INTER_CUBIC);
          cv::resize(imgV, imgV, cv::Size(width, height), 0, 0, INTER_CUBIC);
        }

        float min_u_f, max_u_f, min_v_f, max_v_f;
        if (clipFlow) {
          min_u_f = -20;
          max_u_f = 20;
          min_v_f = -20;
          max_v_f = 20;
        } else {
          double min, max;
          cv::minMaxLoc(imgU, &min, &max);
          min_u_f = min;
          max_u_f = max;
          cv::minMaxLoc(imgV, &min, &max);
          min_v_f = min;
          max_v_f = max;
        }

        // Safe in-place conversion
        convertFlowToImage(imgU, imgU, min_u_f, max_u_f);
        convertFlowToImage(imgV, imgV, min_v_f, max_v_f);

        // Output OF in grey image
        std::ostringstream strStream;
        strStream << "/" << FRAME_FILE_PREFIX << nframes << FRAME_FILE_EXT;
        imwrite(outfile_u + strStream.str(), imgU);
        imwrite(outfile_v + strStream.str(), imgU);
      }

      frame1_rgb.copyTo(frame0_rgb);
      frame1_grey.copyTo(frame0_grey);
      frame1_grey_32f.copyTo(frame0_grey_32f);

      nframes++;
      // TODO: Have a proper skip loop
      for (int iskip = 0; iskip < frameSkip; ++iskip) {
        cap >> frame_iter;  // is it the only way to iterate?
      }

      t2fr = cv::getTickCount();
      tdframe = 1000.0 * (t2fr - t1fr) / cv::getTickFrequency();
      avg_tdframe = avg_tdframe + (tdframe - avg_tdframe) / nframes;
    }
    cout << "Processed video: " << fName << " total frames: " << nframes
         << endl;
    cout << "Average OF compute time per frame pair: " << avg_tdflow << " ms"
         << endl;
    cout << "Average time per frame pair:  " << avg_tdframe << " ms" << endl;
  }
  return 0;
}
