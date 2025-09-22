from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# 인증
gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # 브라우저에서 로그인
drive = GoogleDrive(gauth)

# 업로드할 파일 지정
file_path = "test_image.jpg"  # 로컬 사진
gfile = drive.CreateFile({'title': 'test_image.jpg'})
gfile.SetContentFile(file_path)
gfile.Upload()
print("Drive에 업로드 완료:", gfile['title'])
