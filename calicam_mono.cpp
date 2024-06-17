
// This is NWNC (No Warranty No Copyright) Software.
// astar.ai
// Nov 17, 2019

#include <opencv2/opencv.hpp>

#define MARGIN    15. * CV_PI / 180.

////////////////////////////////////////////////////////////////////////////////

bool      live = false;
//bool      live = true;
//To run live mode, you need a CaliCam from www.astar.ai

int       vfov_bar = 60, size_bar =   0;
int       vfov_max = 60, size_max = 480;
int       vfov_now = 120, size_now = 480;

int       cap_cols, cap_rows;
bool      changed = false;
cv::Mat   Kl, Dl, xil, fmap[2];

////////////////////////////////////////////////////////////////////////////////

enum RectMode {
  RECT_PERSPECTIVE,
  RECT_CYLINDRICAL,
  RECT_FISHEYE,
  RECT_LONGLAT
};

RectMode mode = RECT_PERSPECTIVE;

////////////////////////////////////////////////////////////////////////////////

void OnTrackAngle(int value, void*) {
  vfov_bar = value;
  vfov_now = 60 + vfov_bar;
  changed = true;
}

////////////////////////////////////////////////////////////////////////////////

void OnTrackSize(int value, void*) {
  size_bar = value;
  size_now = 480 + size_bar;
  if (size_now % 2 == 1)
    size_now--;
  changed = true;
}

////////////////////////////////////////////////////////////////////////////////

void LoadParameters(std::string file_name) {
  cv::FileStorage fs(file_name, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    std::cout << "Failed to open ini parameters" << std::endl;
    exit(-1);
  }

  cv::Size cap_size;
  fs["cap_size" ] >> cap_size;
  fs["Kl"       ] >> Kl;
  fs["Dl"       ] >> Dl;
  fs["xil"      ] >> xil;

  fs.release();

  cap_cols  = cap_size.width;
  cap_rows  = cap_size.height;
}

////////////////////////////////////////////////////////////////////////////////

inline double MatRowMul(cv::Mat m, double x, double y, double z, int r) {
  return m.at<double>(r,0) * x + m.at<double>(r,1) * y + m.at<double>(r,2) * z;
}

////////////////////////////////////////////////////////////////////////////////

void InitRectifyMap(cv::Mat K,
                    cv::Mat D,
                    cv::Mat Knew,
                    double xi,
                    cv::Size size,
                    RectMode mode,
                    cv::Mat& map1,
                    cv::Mat& map2) {
  map1.create(size, CV_32F);
  map2.create(size, CV_32F);

  double fx = K.at<double>(0,0);
  double fy = K.at<double>(1,1);
  double cx = K.at<double>(0,2);
  double cy = K.at<double>(1,2);
  double s  = K.at<double>(0,1);

  double k1 = D.at<double>(0,0);
  double k2 = D.at<double>(0,1);
  double p1 = D.at<double>(0,2);
  double p2 = D.at<double>(0,3);

  cv::Mat Ki  = Knew.inv();

  for (int r = 0; r < size.height; ++r) {
    for (int c = 0; c < size.width; ++c) {
      double xc = 0.;
      double yc = 0.;
      double zc = 0.;

      if (mode == RECT_PERSPECTIVE) {
        xc = MatRowMul(Ki, c, r, 1., 0);
        yc = MatRowMul(Ki, c, r, 1., 1);
        zc = MatRowMul(Ki, c, r, 1., 2);
      }

      if (mode == RECT_CYLINDRICAL) {
        double tt = MatRowMul(Ki, c, r, 1., 0);
        double pp = MatRowMul(Ki, c, r, 1., 1) + MARGIN;

        xc = -sin(pp) * cos(tt);
        yc = -cos(pp);
        zc =  sin(pp) * sin(tt);
      }

      if (mode == RECT_FISHEYE) {
        if (hypot(c - Knew.at<double>(0, 2), r - Knew.at<double>(1, 2))
            > Knew.at<double>(0, 0)) {
          map1.at<float>(r,c) = -1.f;
          map2.at<float>(r,c) = -1.f;
          continue;
        }
        double ee = MatRowMul(Ki, c, r, 1., 0);
        double ff = MatRowMul(Ki, c, r, 1., 1);

        double zz = 2. / (ee * ee + ff * ff + 1.);

        xc = zz * ee;
        yc = zz * ff;
        zc = zz - 1.;
      }

      if (mode == RECT_LONGLAT) {
        double tt = MatRowMul(Ki, c, r, 1., 0);
        double pp = MatRowMul(Ki, c, r, 1., 1);

        xc = -cos(tt);
        yc = -sin(tt) * cos(pp);
        zc =  sin(tt) * sin(pp);
      }

      double rr = sqrt(xc * xc + yc * yc + zc * zc);
      double xs = xc / rr;
      double ys = yc / rr;
      double zs = zc / rr;

      double xu = xs / (zs + xi);
      double yu = ys / (zs + xi);

      double r2 = xu * xu + yu * yu;
      double r4 = r2 * r2;
      double xd = (1+k1*r2+k2*r4)*xu + 2*p1*xu*yu + p2*(r2+2*xu*xu);
      double yd = (1+k1*r2+k2*r4)*yu + 2*p2*xu*yu + p1*(r2+2*yu*yu);

      double u = fx * xd + s * yd + cx;
      double v = fy * yd + cy;

      map1.at<float>(r,c) = (float) u;
      map2.at<float>(r,c) = (float) v;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

void InitRectifyMap() {
  double   vfov_rad, focal;
  cv::Mat  Knew;
  cv::Size img_size(size_now, size_now);

  switch (mode) {
  case RECT_PERSPECTIVE:
    std::cout << "\x1b[1;36m" << "Mode: " << "Perspective" << "\x1b[0m\n";
    vfov_rad = vfov_now * CV_PI / 180.;
    focal = size_now / 2. / tan(vfov_rad / 2.);
    Knew = (cv::Mat_<double>(3, 3) << focal, 0., size_now  / 2. - 0.5,
                                      0., focal, size_now / 2. - 0.5,
                                      0., 0., 1.);
    InitRectifyMap(Kl, Dl, Knew, xil.at<double>(0,0),
                   img_size, RECT_PERSPECTIVE, fmap[0], fmap[1]);

    std::cout << "V.Fov: "  << vfov_now   << "\t";
    break;

  case RECT_CYLINDRICAL:
    std::cout << "\x1b[1;36m" << "Mode: " << "Cylindrical" << "\x1b[0m\n";
    Knew = cv::Mat::eye(3, 3, CV_64F);
    Knew.at<double>(0,0) = size_now  /  CV_PI;
    Knew.at<double>(1,1) = size_now / (CV_PI - 2 * MARGIN);
    InitRectifyMap(Kl, Dl, Knew, xil.at<double>(0,0),
                   img_size, RECT_CYLINDRICAL, fmap[0], fmap[1]);
    break;

  case RECT_FISHEYE:
    img_size = cv::Size(size_now, size_now);

    std::cout << "\x1b[1;36m" << "Mode: " << "Fisheye" << "\x1b[0m\n";
    Knew = cv::Mat::eye(3, 3, CV_64F);
    Knew.at<double>(0,0) = size_now  / 2.;
    Knew.at<double>(0,2) = size_now  / 2. - 0.5;
    Knew.at<double>(1,1) = size_now  / 2.;
    Knew.at<double>(1,2) = size_now / 2. - 0.5;
    InitRectifyMap(Kl, Dl, Knew, xil.at<double>(0,0),
                   img_size, RECT_FISHEYE, fmap[0], fmap[1]);
    break;

  case RECT_LONGLAT:
    std::cout << "\x1b[1;36m" << "Mode: " << "Longitude-Latitude" << "\x1b[0m\n";
    Knew = cv::Mat::eye(3, 3, CV_64F);
    Knew.at<double>(0,0) = size_now  / CV_PI;
    Knew.at<double>(1,1) = size_now / CV_PI;
    InitRectifyMap(Kl, Dl, Knew, xil.at<double>(0,0),
                   img_size, RECT_LONGLAT, fmap[0], fmap[1]);
    break;
  }

  std::cout << "Width: "  << size_now  << "\t"
            << "Height: " << size_now << std::endl;
  std::cout << "K Matrix: \n" << Knew << std::endl << std::endl;
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
  std::string param_name = "../astar_calicam_mono.yml";
  std::string image_name = "../times_square.jpg";

  if (argc == 2) {
    param_name = argv[1];
  } else if (argc == 3) {
    param_name = argv[1];
    image_name = argv[2];
  }

  LoadParameters(param_name);
  InitRectifyMap();

  cv::Mat raw_img;
  cv::VideoCapture vcapture;
  if (live) {
    vcapture.open(0);

    if (!vcapture.isOpened()) {
      std::cout << "Camera doesn't work" << std::endl;
      exit(-1);
    }

    vcapture.set(cv::CAP_PROP_FRAME_WIDTH,  cap_cols);
    vcapture.set(cv::CAP_PROP_FRAME_HEIGHT, cap_rows);
    vcapture.set(cv::CAP_PROP_FPS, 30);
  } else {
    raw_img = cv::imread(image_name, cv::IMREAD_COLOR);
  }

  char win_name[256];
  sprintf(win_name, "Raw Image: %d x %d", cap_cols, cap_rows);
  std::string param_win_name(win_name);
  cv::namedWindow(param_win_name);

  cv::createTrackbar("V. FoV:  60    +", param_win_name, nullptr,   vfov_max,   OnTrackAngle);
  cv::setTrackbarPos("V. FoV:  60    +", param_win_name, vfov_bar);
  cv::createTrackbar("Size:  480 +", param_win_name, nullptr,  size_max,  OnTrackSize);

  cv::Mat raw_imgl, raw_imgr, rect_imgl, rect_imgr;
  while (1) {
    if (changed) {
      InitRectifyMap();
      changed = false;
    }

    if (live)
      vcapture >> raw_img;

    if (raw_img.total() == 0) {
      std::cout << "Image capture error" << std::endl;
      exit(-1);
    }

    raw_imgl = raw_img;
    cv::remap(raw_img, rect_imgl, fmap[0], fmap[1], 1, 0);

    cv::Mat small_img;
    cv::resize(raw_imgl, small_img, cv::Size(), 0.5, 0.5);
    imshow(param_win_name, small_img);
    imshow("Rectified Image", rect_imgl);

    char key = cv::waitKey(1);

    if (key == '1') {
      mode = RECT_PERSPECTIVE;
      changed = true;
    }

    if (key == '2') {
      mode = RECT_CYLINDRICAL;
      changed = true;
    }

    if (key == '3') {
      mode = RECT_FISHEYE;
      changed = true;
    }

    if (key == '4') {
      mode = RECT_LONGLAT;
      changed = true;
    }

    if (key == 32) {
      imwrite("img.jpg", rect_imgl);
    }

    if (key == 'q' || key == 'Q' || key == 27)
      break;
  }

  return 0;
}

////////////////////////////////////////////////////////////////////////////////
