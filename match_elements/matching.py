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


def compare_sift_or_surf(img1, img2, method, ratio=1.5, draw_match=False):
    if method == 'sift':
        sift = cv2.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
    elif method == 'surf':
        # Initiate SURF detector
        surf = cv2.xfeatures2d.SURF_create()
        # find the keypoints and descriptors with SURF
        kp1, des1 = surf.detectAndCompute(img1, None)
        kp2, des2 = surf.detectAndCompute(img2, None)
    else:
        print('set method as either sift or surf')
        return

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # If there's a big difference between the best and second-best matches, this to be a quality match.
    valid_matches = []
    for best, second in matches:
        if second.distance / best.distance > ratio:
            valid_matches.append(best)

    if draw_match:
        board = cv2.drawMatches(img1, kp1, img2, kp2, valid_matches, None)
        cv2.imshow('sift match', cv2.resize(board, (int(board.shape[1] * (200 / board.shape[0])), 200)))
        cv2.waitKey()
        cv2.destroyWindow('sift match')
    return len(valid_matches) / len(kp2)


def image_similarity(img1, img2, method='dhash', is_gray=False,
                     draw_match=False, match_distance_ratio=1.5):
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
    elif method == 'sift':
        similarity = compare_sift_or_surf(img1, img2, 'sift', match_distance_ratio, draw_match=draw_match)
    elif method == 'surf':
        similarity = compare_sift_or_surf(img1, img2, 'surf', match_distance_ratio, draw_match=draw_match)
    return similarity


def text_similarity(text1, text2):
    return SequenceMatcher(None, text1, text2).ratio()
