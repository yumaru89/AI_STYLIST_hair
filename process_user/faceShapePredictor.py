import os
import sys
import pickle

import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.facemark import facemark
from utils.prepare_mask import prepare_hair_mask, prepare_network
from utils.removeBackground import run_visualization
from utils.utils import has_img_ext, make_same_size


def main(img_path, model_path):
    user_face = cv2.imread(img_path)
    user_face_gray = cv2.cvtColor(user_face, cv2.COLOR_BGR2GRAY)
    user_lms = facemark(user_face_gray)

    if len(user_lms) > 1: return print("複数人検出")
    if not len(user_lms): return print("検出不可")

    user_lm = user_lms[0]

    # モデル髪がユーザーの顔に合うように位置調整
    model_hair = apply_user_face(model_path, user_lm)

    # ユーザーの顔から、髪の毛と背景の削除
    user_face = remove_hair(user_face, img_path, user_lm, model_hair)
    user_new_face = run_visualization(user_face)
    user_new_face = cv2.cvtColor(user_new_face, cv2.COLOR_RGBA2BGRA)
    user_new_face = cv2.resize(user_new_face, (user_face.shape[1], user_face.shape[0]))

    save_dir = './test/face'
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(f"test/face/{model_path}.png", user_new_face)
    save_dir = './test/hair'
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(f"test/hair/{model_path}.png", model_hair)

    cv2.destroyAllWindows()


def apply_user_face(model_path, user_lm):
    model_hair = cv2.imread(f"img/model/hair/{model_path}.png", flags=cv2.IMREAD_UNCHANGED)

    with open("model_data/model_shape.pickle", mode="rb") as f:
        models_shape = pickle.load(f)

    model_lm = models_shape[model_path]
    model_lm = np.array(model_lm)


    user_vec = user_lm[0] - user_lm[16]

    model_vec = model_lm[0] - model_lm[16]
    hair_ratio = np.linalg.norm(user_vec) / np.linalg.norm(model_vec)
    hair_h_ratio = (user_lm[71,0] - user_lm[19,0]) / (model_lm[71,0] - model_lm[19,0])

    model_uint_lm = model_lm.astype(np.int64)

    # モデル髪の縦横のサイズ調整
    model_hair = cv2.resize(model_hair, (int(np.round(model_hair.shape[1]*hair_ratio)), int(np.round(model_hair.shape[0]*hair_ratio))))
    model_lm = model_lm*hair_ratio
    # モデル髪の縦のサイズ調整
    model_top = model_hair[:model_uint_lm[71,1]]
    model_middle = model_hair[model_uint_lm[71,1]:model_uint_lm[19,1]]
    model_bottom = model_hair[model_uint_lm[19,1]:]
    n_model_middle = cv2.resize(model_middle, (model_hair.shape[1], int(np.round(model_middle.shape[0] * hair_h_ratio))))
    model_hair = np.append(model_top, n_model_middle, axis=0)
    model_hair = np.append(model_hair, model_bottom, axis=0)

    # landmarkの辻褄合わせ
    diff_model_middle = model_middle.shape[0] - n_model_middle.shape[0]
    model_lm[:,1] = np.where((model_lm[:,1] > model_lm[71,1])&(model_lm[:,1] <= model_lm[19,1]), model_lm[71,1]+((model_lm[:,1]-model_lm[71,1])*hair_h_ratio), model_lm[:, 1])
    model_lm[:,1] = np.where(model_lm[:,1] > model_lm[19,1], model_lm[:,1]-diff_model_middle, model_lm[:,1])

    # landmarkを変えたから計算し直し
    axis_point_diff = user_lm[0] - model_lm[0]
    user_vec = user_lm[0] - user_lm[16]
    model_vec = model_lm[0] - model_lm[16]

    inner = np.inner(user_vec, model_vec)
    norm = np.linalg.norm(user_vec) * np.linalg.norm(model_vec)
    hair_rad = np.arccos(np.clip(inner/norm, -1.0, 1.0))


    # モデル髪とユーザーの基準点が会うように移動
    h, w = model_hair.shape[:2]
    src = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], np.float32)
    dest = src + axis_point_diff.reshape(1, -1).astype(np.float32)
    affine = cv2.getAffineTransform(src, dest)
    model_hair = cv2.warpAffine(model_hair, affine, (w, h))

    # ユーザーの基準点を中心として回転
    center = (user_lm[0,0], user_lm[0,1])
    angle = np.rad2deg(hair_rad)
    if model_vec[1] > user_vec[1]:
        trans_hair = cv2.getRotationMatrix2D(center, -angle, 1)
    else:
        trans_hair = cv2.getRotationMatrix2D(center, angle, 1)
    model_hair = cv2.warpAffine(model_hair, trans_hair, (w, h))

    return model_hair


def remove_hair(user_face, img_path, face_lm, model_hair):
    adjust_lm = np.stack([face_lm[78], face_lm[74], face_lm[79], face_lm[73], face_lm[72], face_lm[80], face_lm[71], face_lm[70], face_lm[69], face_lm[68], face_lm[76], face_lm[75], face_lm[77]])
    face_clm = np.append(face_lm[:16], adjust_lm, axis=0)
    face_mask = np.zeros(user_face.shape[:2], np.uint8)
    cv2.drawContours(face_mask, [face_clm], -1, (255, 255, 255), -1, cv2.LINE_AA)

    face_mask_rgba = cv2.cvtColor(face_mask, cv2.COLOR_BGR2RGBA)
    face_mask_rgba[:,:,3] = np.where(face_mask_rgba[:,:,0]==255, 255, 0)

    test_image_transforms, net = prepare_network()
    hair_mask = prepare_hair_mask(img_path, test_image_transforms, net)

    mask_hair_remove = cv2.bitwise_not(hair_mask)
    mask_hair_remove = cv2.cvtColor(mask_hair_remove, cv2.COLOR_GRAY2RGBA)

    face_mask_nohair = cv2.addWeighted(mask_hair_remove, 1, face_mask_rgba, 1, 0)
    face_mask_nohair = cv2.cvtColor(face_mask_nohair, cv2.COLOR_BGRA2GRAY)

    user_face_bgr = cv2.split(user_face)
    user_nohair_face = cv2.merge(user_face_bgr + [face_mask_nohair])

    # 合成後に透過する場所を埋める Gだけ255に変える(0,255,0 緑)
    face_fill_mask = user_face_fill(model_hair, user_nohair_face, face_lm)
    face_fill_mask = cv2.cvtColor(face_fill_mask, cv2.COLOR_GRAY2BGR)
    face_fill_mask[:,:,1] = np.where(face_fill_mask[:,:,1]==0, 255, face_fill_mask[:,:,1])
    # 髪の毛のマスクのBだけ255に変える(255,0,0 青)
    face_mask_nohair = cv2.cvtColor(face_mask_nohair, cv2.COLOR_GRAY2BGR)
    face_mask_nohair[:,:,0] = np.where(face_mask_nohair[:,:,0]==0, 255, face_mask_nohair[:,:,0])
    # 合成する 元のマスク部分は、255,128,128  埋める部分は、128, 128, 0
    face_fill_mask, face_mask_nohair = make_same_size(face_fill_mask, face_mask_nohair)
    face_mask_nohair_fill = cv2.addWeighted(face_mask_nohair, 0.5, face_fill_mask, 0.5, 0)
    # Rが128のところを(0,0,0 黒)にする
    for i in range(3):
        face_mask_nohair_fill[:,:,i] = np.where(face_mask_nohair_fill[:,:,2]==128, 0, 255)

    face_mask_nohair_fill = face_mask_nohair_fill[:,:,:1]
    # face_mask_nohairをalpha値として合成
    face_mask_nohair_fill, user_face = make_same_size(face_mask_nohair_fill, user_face)
    user_face_bgr = cv2.split(user_face)
    user_nohair_face = cv2.merge(user_face_bgr + [face_mask_nohair_fill])

    return user_nohair_face


def user_face_fill(model_hair, user_face, user_lm):
    # 合成するために、画像のサイズを合わせる 拡大すると崩れるから、透明なndarrayを入れて調節
    user_face = cv2.cvtColor(user_face, cv2.COLOR_BGR2RGBA)
    model_hair, user_face = make_same_size(model_hair, user_face)
    new_user_face = cv2.addWeighted(user_face, 1, model_hair, 1, 0)
    new_user_face_alpha = new_user_face[:,:,3:]
    mask_img = np.where(new_user_face_alpha<200, 255, 0)

    face_fill = mask_img[user_lm[69,1]-20:user_lm[69,1], user_lm[69,0]:user_lm[72,0]]
    face_fill_mask = np.zeros((new_user_face.shape[0], new_user_face.shape[1], 1), dtype=np.uint8)
    face_fill_mask[user_lm[69,1]-20:user_lm[69,1], user_lm[69,0]:user_lm[72,0]] = face_fill
    face_fill_mask[:] = np.where(face_fill_mask[:] == 255,  0, 255)

    return face_fill_mask


def drawn_landmark(user_face, face_lm):
    for i, points in enumerate(np.append(face_lm[:16], face_lm[68:80], axis=0)):
        cv2.drawMarker(user_face, (points[0], points[1]), (255, 21, 12), markerSize=5)

    return user_face

if __name__ == '__main__':
    hair_dir = "./img/model/hair"
    hair_paths = [os.path.join(hair_dir, k) for k in sorted(os.listdir(hair_dir)) if has_img_ext(k)]
    hair_names = [os.path.splitext(os.path.basename(img_path))[0] for img_path in hair_paths]
    # print(hair_names)
    for hair_name in hair_names:
        try:
            main("img/user/resize_face/face5.png", hair_name)
        except IndexError:
            print(f"顔が複数orなし: {hair_name}")
            continue
        except cv2.error:
            print(f"cv2.error: {hair_name}")
            continue
    # main("img/user/resize_face/face5.png", "07f37efa-32cc-11ea-9997-dca904866664_000")

