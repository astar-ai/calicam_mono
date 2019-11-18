g++ -O3 -Wall -c -fmessage-length=0 -MMD -MP -MF"calicam_mono.d" -MT"calicam_mono.o" -o "calicam_mono.o" "calicam_mono.cpp"
g++  -o "calicam_mono"  ./calicam_mono.o   -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio -lopencv_calib3d
rm *.d *.o

