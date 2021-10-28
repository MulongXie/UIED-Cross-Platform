import cv2


def dhash(image):
    image = cv2.resize(image, (9, 8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
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


def comp_similarity(img1, img2, method='dhash'):
    similarity = -1  # from 0 to 1
    if method == 'dhash':
        h1 = dhash(img1)
        h2 = dhash(img2)
        similarity = compare_hash(h1, h2)
    return similarity
