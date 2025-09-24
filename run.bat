@echo off
REM 人脸颜值评分程序运行脚本
cd /d "g:\zizim\Documents\code\python\AI\lab_1\src\ujn_AI_class_lab_1"

echo 人脸颜值评分程序
echo ===================
echo 1. 摄像头实时评分 (默认)
echo 2. 图片评分
echo 3. 视频评分
echo.

set /p choice="请选择运行模式 (1-3，默认1): "

if "%choice%"=="2" (
    set /p image_path="请输入图片路径: "
    C:\Users\zizim\.julia\conda\3\x86_64\Scripts\conda.exe run -p C:\Users\zizim\.julia\conda\3\x86_64 --no-capture-output python sample.py --image "%image_path%"
) else if "%choice%"=="3" (
    set /p video_path="请输入视频路径: "
    C:\Users\zizim\.julia\conda\3\x86_64\Scripts\conda.exe run -p C:\Users\zizim\.julia\conda\3\x86_64 --no-capture-output python sample.py --video "%video_path%"
) else (
    echo 启动摄像头实时颜值评分...
    C:\Users\zizim\.julia\conda\3\x86_64\Scripts\conda.exe run -p C:\Users\zizim\.julia\conda\3\x86_64 --no-capture-output python sample.py --camera 0
)

pause