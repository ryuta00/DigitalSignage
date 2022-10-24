from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import uuid
import cv2

from users.models import News, Reaction, Student
from users.forms import ImageUploadForm
from django.conf import settings


img_size = 160


# 生徒の顔写真データを保存する
def save_vector(vector):
    name = str(uuid.uuid4())
    np.save(
        "users/students/" + name + ".npy",
        vector
    )
    return "users/students/" + name + ".npy"


# 生徒の顔写真データを取得
def get_vector(student):
    return np.load(student.vector_path)


# 顔写真をベクトル化
def parse_vector(image_path):
    mtcnn = MTCNN(image_size=160, margin=10)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    return resnet(mtcnn(Image.open(image_path)).unsqueeze(0)).squeeze().to('cpu').detach().numpy().copy()


# 類似度の関数
def cos_similarity(p1, p2):
    return np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))


def login(request):
    return render(request, 'login.html')


def signup(request):
    if request.method == 'POST':
        # ユーザー顔写真の保存
        form = ImageUploadForm(request.POST,  request.FILES)
        upload_image = settings.MEDIA_ROOT + form.save()

        # ユーザー登録
        Student.objects.create(
            name=request.POST["name"],
            mail=request.POST["mail"],
            vector_path=save_vector(parse_vector(upload_image))
        )
        return HttpResponse(upload_image)

    form = ImageUploadForm
    return render(request, 'signup.html', {"form": form})


@login_required
def home(request):
    return render(request, 'home.html')


@login_required
def news(request):
    if request.method == 'POST':
        Reaction.objects.create(
            user=request.user,
            stamp=request.POST['stamp'],
            news=News.objects.get(id=request.POST['news'])
        )
        return redirect("news")

    else:
        ctx = {
            "news_list": News.objects.all()
        }
        return render(request, 'news.html', ctx)


def match(request):
    students = Student.objects.all()
    students_data = {}
    for student in students:
        students_data[student.id] = get_vector(student)

    # If required, create a face detection pipeline using MTCNN:
    mtcnn = MTCNN(image_size=160, margin=10)

    # Create an inception resnet (in eval mode):
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    cap = cv2.VideoCapture(0)

    user_id = ""
    score = 0.0
    check_count = 0.0
    while True:
        ret, frame = cap.read()
        cv2.imshow('Face Match', frame)

        k = cv2.waitKey(10)
        if k == 27:
            break

        # numpy to PIL
        img_cam = Image.fromarray(frame)
        img_cam = img_cam.resize((img_size, img_size))

        img_cam_cropped = mtcnn(img_cam)
        if img_cam_cropped is not None:
            # if len(img_cam_cropped.size()) != 0:
            img_embedding = resnet(img_cam_cropped.unsqueeze(0))

            x2 = img_embedding.squeeze().to('cpu').detach().numpy().copy()

            user_id = ""
            score = 0.0
            for key in students_data:
                x1 = students_data[key]
                print("--------------------")
                print(key)
                print(cos_similarity(x1, x2))
                print("---------------------")
                if cos_similarity(x1, x2) > 0.7:
                    user_id = key
                    score = cos_similarity(x1, x2)
                    break

            check_count += 1.0
        else:
            check_count += 0.2

        # 色々ループ抜ける条件
        if user_id:
            break

        if check_count > 20:
            break

    cap.release()
    cv2.destroyAllWindows()

    if user_id:
        print("ここ")
        return HttpResponse('顔認証 : ' + Student.objects.get(id=user_id).name)
    else:
        print("here")
        return HttpResponse('顔認証 : 該当するユーザーがいません！')


def face(request):
    return render(request, 'match.html')


def start(request):
    # 顔検出のAI
    # image_size: 顔を検出して切り取るサイズ
    # margin: 顔まわりの余白
    mtcnn = MTCNN(image_size=160, margin=10)

    # 切り取った顔を512個の数字にするAI
    # 1回目の実行では学習済みのモデルをダウンロードしますので、少し時間かかります。
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    # 三人分の比較をします。
    # 1つ目をカメラで取得した人として
    # 2、3つ目を登録されている人とします。
    image_path = settings.STATIC_ROOT + "/images/face.jpg"
    image_path1 = settings.STATIC_ROOT + "/images/face.jpg"
    image_path2 = settings.STATIC_ROOT + "/images/face2.jpg"
    image_path3 = settings.STATIC_ROOT + "/images/face3.jpg"

    img0 = Image.open(image_path2)
    img_cropped0 = mtcnn(img0)
    img_embedding0 = resnet(img_cropped0.unsqueeze(0))

    # (仮)カメラで取得した方
    # 画像データ取得
    img1 = Image.open(image_path1)
    # 顔データを160×160に切り抜き
    img_cropped1 = mtcnn(img1)
    # save_pathを指定すると、切り取った顔画像が確認できます。
    # img_cropped1 = mtcnn(img1, save_path="cropped_img1.jpg")
    # 切り抜いた顔データを512個の数字に
    img_embedding1 = resnet(img_cropped1.unsqueeze(0))

    # (仮)登録されたカメラと同じ人
    img2 = Image.open(image_path2)
    img_cropped2 = mtcnn(img2)
    img_embedding2 = resnet(img_cropped2.unsqueeze(0))

    # (仮)登録されたカメラと違う人
    img3 = Image.open(image_path3)
    img_cropped3 = mtcnn(img3)
    img_embedding3 = resnet(img_cropped3.unsqueeze(0))

    # 512個の数字にしたものはpytorchのtensorという型なので、numpyの方に変換
    p0 = img_embedding0.squeeze().to('cpu').detach().numpy().copy()
    p1 = img_embedding1.squeeze().to('cpu').detach().numpy().copy()
    p2 = img_embedding2.squeeze().to('cpu').detach().numpy().copy()
    p3 = img_embedding3.squeeze().to('cpu').detach().numpy().copy()

    # 類似度を計算して顔認証
    img1vs2 = cos_similarity(p1, p2)
    img1vs3 = cos_similarity(p1, p3)

    # ユーザー登録
    Student.objects.create(
        name="野口龍太A",
        mail="s1f101900936@iniad.org",
        vector_path=save_vector(p0)
    )
    Student.objects.create(
        name="野口龍太B",
        mail="s1f101900936@iniad.org",
        vector_path=save_vector(p1)
    )
    Student.objects.create(
        name="野口龍太C",
        mail="s1f101900936@iniad.org",
        vector_path=save_vector(p2)
    )
    Student.objects.create(
        name="野口龍D",
        mail="s1f101900936@iniad.org",
        vector_path=save_vector(p3)
    )

    return HttpResponse(p1)
