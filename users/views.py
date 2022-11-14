from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import uuid
import cv2

from users.models import News, Reaction, Student, StudentReaction
from users.forms import ImageUploadForm
from django.conf import settings

# 顔認証用変数
img_size = 160

# 姿勢推定のモデルダウンロード
model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']
KEYPOINT_THRESHOLD = 0.2


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


# ニュースを表示
# ニュースを表示したらユーザー認証を開始
def news(request, news):
    ctx = {
        "news": News.objects.get(id=news)
    }
    return render(request, 'news.html', ctx)


# ユーザー認証後もニュースを表示
# ユーザーのリアクションを測定開始
def reaction(request, news, user):
    ctx = {
        "news": News.objects.get(id=news),
        "user": Student.objects.get(id=user)
    }
    return render(request, 'reaction.html', ctx)


# # @login_required
# def index(request):
#     if request.method == 'POST':
#         Reaction.objects.create(
#             user=request.user,
#             stamp=request.POST['stamp'],
#             news=News.objects.get(id=request.POST['news'])
#         )
#         return redirect("index")
#
#     else:
#         ctx = {
#             "news_list": News.objects.all()
#         }
#         return render(request, 'index.html', ctx)


def camera(request, news, user):
    # カメラ設定
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    rightpoint = 0
    leftpoint = 0
    checkcount = 0

    while (checkcount <= 10000):
        ret, frame = cap.read()
        if not ret:
            break

        # 推論実行
        keypoints_list, scores_list, bbox_list = run_inference(movenet, frame)

        # 画像レンダリング
        result_image, rightpoint, leftpoint = myrender(frame, keypoints_list, scores_list, bbox_list, rightpoint, leftpoint)

        if rightpoint >= 30:
            print("右手だーーーーーーー")
            StudentReaction.objects.create(
                user=Student.objects.get(id=user),
                stamp=1,
                news=News.objects.get(id=news)
            )
            ctx = {
                "user": Student.objects.get(id=user),
                "reaction": "ok"
            }
            return render(request, 'result.html', ctx)
        if leftpoint >= 30:
            print("左手だーーーーーーー")
            StudentReaction.objects.create(
                user=Student.objects.get(id=user),
                stamp=0,
                news=News.objects.get(id=news)
            )
            ctx = {
                "user": Student.objects.get(id=user),
                "reaction": "no"
            }
            return render(request, 'result.html', ctx)
        checkcount += 1

    return redirect("reaction", news=news, user=user)


def run_inference(model, image):
    # 画像の前処理
    input_image = cv2.resize(image, dsize=(256, 256))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = np.expand_dims(input_image, 0)
    input_image = tf.cast(input_image, dtype=tf.int32)

    # 推論実行・結果取得
    outputs = model(input_image)
    keypoints = np.squeeze(outputs['output_0'].numpy())

    image_height, image_width = image.shape[:2]
    keypoints_list, scores_list, bbox_list = [], [], []

    # 検出した人物ごとにキーポイントのフォーマット処理
    for kp in keypoints:
        keypoints = []
        scores = []
        for index in range(17):
            kp_x = int(image_width * kp[index * 3 + 1])
            kp_y = int(image_height * kp[index * 3 + 0])
            score = kp[index * 3 + 2]
            keypoints.append([kp_x, kp_y])
            scores.append(score)
        bbox_ymin = int(image_height * kp[51])
        bbox_xmin = int(image_width * kp[52])
        bbox_ymax = int(image_height * kp[53])
        bbox_xmax = int(image_width * kp[54])
        bbox_score = kp[55]

        keypoints_list.append(keypoints)
        scores_list.append(scores)
        bbox_list.append([bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, bbox_score])

    return keypoints_list, scores_list, bbox_list


def myrender(image, keypoints_list, scores_list, bbox_list, rightpoint, leftpoint):
    render = image.copy()
    for i, (keypoints, scores, bbox) in enumerate(zip(keypoints_list, scores_list, bbox_list)):
        if bbox[4] < 0.2:
            continue

        cv2.rectangle(render, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        # 0:nose, 1:left eye, 2:right eye, 3:left ear, 4:right ear, 5:left shoulder, 6:right shoulder, 7:left elbow, 8:right elbow, 9:left wrist, 10:right wrist,
        # 11:left hip, 12:right hip, 13:left knee, 14:right knee, 15:left ankle, 16:right ankle
        # 接続するキーポイントの組
        kp_links = [
            (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6), (5, 7), (7, 9), (6, 8),
            (8, 10), (11, 12), (5, 11), (11, 13), (13, 15), (6, 12), (12, 14), (14, 16)
        ]
        for kp_idx_1, kp_idx_2 in kp_links:
            kp_1 = keypoints[kp_idx_1]
            kp_2 = keypoints[kp_idx_2]
            score_1 = scores[kp_idx_1]
            score_2 = scores[kp_idx_2]
            if score_1 > KEYPOINT_THRESHOLD and score_2 > KEYPOINT_THRESHOLD:
                cv2.line(render, tuple(kp_1), tuple(kp_2), (0, 0, 255), 2)

        for idx, (keypoint, score) in enumerate(zip(keypoints, scores)):
            if score > KEYPOINT_THRESHOLD:
                cv2.circle(render, tuple(keypoint), 4, (0, 0, 255), -1)

        if keypoints[10][1] <= keypoints[6][1]:
            print("----------------------\n \n \n 挙手：右")
            rightpoint += 1
            print(keypoints[10])

        if keypoints[9][1] <= keypoints[5][1]:
            print("----------------------\n \n \n 挙手：左")
            leftpoint += 1
            print(keypoints[9])

    return render, rightpoint, leftpoint


# ユーザー認証
def match(request, news):
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
        if k == 100:
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
        print("ログイン：" + Student.objects.get(id=user_id).name)
        return redirect("reaction", news=news, user=user_id)
    else:
        print("登録済みユーザーと合致せず。全画面にリダイレクト")
        return redirect("news", news=news)


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
    #image_path2 = settings.STATIC_ROOT + "/images/face2.jpg"
    #image_path3 = settings.STATIC_ROOT + "/images/face3.jpg"

    img0 = Image.open(image_path)
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
    #img2 = Image.open(image_path2)
    #img_cropped2 = mtcnn(img2)
    #img_embedding2 = resnet(img_cropped2.unsqueeze(0))

    # (仮)登録されたカメラと違う人
    #img3 = Image.open(image_path3)
    #img_cropped3 = mtcnn(img3)
    #img_embedding3 = resnet(img_cropped3.unsqueeze(0))

    # 512個の数字にしたものはpytorchのtensorという型なので、numpyの方に変換
    p0 = img_embedding0.squeeze().to('cpu').detach().numpy().copy()
    p1 = img_embedding1.squeeze().to('cpu').detach().numpy().copy()
    #p2 = img_embedding2.squeeze().to('cpu').detach().numpy().copy()
    #p3 = img_embedding3.squeeze().to('cpu').detach().numpy().copy()

    # 類似度を計算して顔認証
    #img1vs2 = cos_similarity(p1, p2)
    #img1vs3 = cos_similarity(p1, p3)

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

    return HttpResponse(p1)
