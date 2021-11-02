import cv2
from difflib import SequenceMatcher
from skimage.measure import compare_ssim


def dhash(image):
    image = cv2.resize(image, (9, 8), interpolation=cv2.INTER_CUBIC)
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    dhash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                dhash_str = dhash_str + '1'
            else:
                dhash_str = dhash_str + '0'
    return dhash_str


def compare_hash(hash1, hash2):
    n = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1
    return 1 - n / len(hash1)


def image_similarity(img1, img2, method='dhash', is_gray=False):
    similarity = -1  # from 0 to 1
    if method == 'dhash':
        h1 = dhash(img1)
        h2 = dhash(img2)
        similarity = compare_hash(h1, h2)
    elif method == 'ssim':
        multi_channel = True
        if is_gray:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            multi_channel = False
        if img1.shape != img2.shape:
            side = max(img1.shape[:2] + img2.shape[:2])
            img1_cp = cv2.resize(img1, (side, side))
            img2_cp = cv2.resize(img2, (side, side))
            similarity = compare_ssim(img1_cp, img2_cp, multichannel=multi_channel)
        else:
            similarity = compare_ssim(img1, img2, multichannel=multi_channel)
    return similarity


def text_similarity(text1, text2):
    return SequenceMatcher(None, text1, text2).ratio()
