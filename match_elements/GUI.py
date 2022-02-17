import cv2
import json
import os
import numpy as np
import time
import shutil
from os.path import join as pjoin
from random import randint as rint
from glob import glob

from match_elements.Element import Element
import match_elements.matching as match
from sklearn.metrics.pairwise import cosine_similarity


class GUI:
    def __init__(self, img_path='data/input', output_dir='data/output', detection_resize_height=900):
        self.img_path = img_path
        self.ui_name = img_path.replace('\\', '/').split('/')[-1].split('.')[0]
        self.output_dir = output_dir

        self.detection_resize_height = detection_resize_height  # resize the input gui while detecting
        self.img = cv2.imread(self.img_path)
        self.det_result_imgs = {'text': None, 'non-text': None, 'merge': None}  # image visualization for different stages
        self.det_result_data = None         # {'compos':[], 'img_shape'}

        self.elements = []                  # list of Element objects for android UI
        self.elements_mapping = {}          # {'id': Element}
        self.has_popup_modal = False        # if the ui has popup modal

    '''
    **********************
    *** GUI Operations ***
    **********************
    '''
    def popup_modal_recognition(self, height_thresh=0.15, width_thresh=0.5):
        def is_element_modal(element, area_resize):
            gray = cv2.cvtColor(element.clip, cv2.COLOR_BGR2GRAY)
            area_ele = element.clip.shape[0] * element.clip.shape[1]
            # calc the grayscale of the element
            sum_gray_ele = np.sum(gray)
            mean_gray_ele = sum_gray_ele / area_ele
            # calc the grayscale of other region except the element
            sum_gray_other = sum_gray_a - sum_gray_ele
            mean_gray_other = sum_gray_other / (area_resize - area_ele)
            # if the element's brightness is far higher than other regions, it should be a pop-up modal
            if mean_gray_ele > 180 and mean_gray_other < 80:
                return True
            return False

        # calculate the mean pixel value as the brightness
        img_resized = cv2.resize(self.img, (int(self.img.shape[1] * (self.detection_resize_height / self.img.shape[0])), self.detection_resize_height))
        area_resize = img_resized.shape[0] * img_resized.shape[1]

        sum_gray_a = np.sum(cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY))

        if sum_gray_a / (img_resized.shape[0] * img_resized.shape[1]) < 100:
            for ele in self.elements:
                if ele.category == 'Compo' and \
                        ele.height / ele.detection_img_size[0] > height_thresh and ele.width / ele.detection_img_size[1] > width_thresh:
                    ele.get_clip(img_resized)
                    if is_element_modal(ele, area_resize):
                        self.has_popup_modal = True
                        ele.is_popup_modal = True
        if not self.has_popup_modal:
            print("No popup modal")

    '''
    *******************************
    *** Detect or Load Elements ***
    *******************************
    '''
    def element_detection(self, is_text=True, is_nontext=True, is_merge=True, paddle_cor=None):
        if is_text:
            os.makedirs(pjoin(self.output_dir, 'ocr'), exist_ok=True)
            import detect_text.text_detection as text
            if not paddle_cor:
                from paddleocr import PaddleOCR
                paddle_cor = PaddleOCR(use_angle_cls=True, lang="ch")
            self.det_result_imgs['text'] = text.text_detection_paddle(self.img_path, pjoin(self.output_dir, 'ocr'), paddle_cor=paddle_cor)
        if is_nontext:
            os.makedirs(pjoin(self.output_dir, 'ip'), exist_ok=True)
            import detect_compo.ip_region_proposal as ip
            key_params = {'min-grad': 6, 'ffl-block': 5, 'min-ele-area': 100, 'merge-contained-ele': False}
            self.det_result_imgs['non-text'] = ip.compo_detection(self.img_path, self.output_dir, key_params, resize_by_height=self.detection_resize_height, adaptive_binarization=False)
        if is_merge:
            os.makedirs(pjoin(self.output_dir, 'merge'), exist_ok=True)
            import detect_merge.merge as merge
            compo_path = pjoin(self.output_dir, 'ip', str(self.ui_name) + '.json')
            ocr_path = pjoin(self.output_dir, 'ocr', str(self.ui_name) + '.json')
            self.det_result_imgs['merge'], self.det_result_data = merge.merge(self.img_path, compo_path, ocr_path, pjoin(self.output_dir, 'merge'), is_remove_bar=True, is_paragraph=False)
            # convert elements as Element objects
            self.cvt_elements()

    def load_detection_result(self, data_path=None):
        if not data_path:
            data_path = pjoin(self.output_dir, 'merge', self.ui_name + '.json')
        self.det_result_data = json.load(open(data_path))
        # convert elements as Element objects
        self.cvt_elements()

    '''
    **************************************
    *** Operations for Element Objects ***
    **************************************
    '''
    def cvt_elements(self):
        '''
        Convert detection result to Element objects
        @ det_result_data: {'elements':[], 'img_shape'}
        '''
        class_map = {'Text': 't', 'Compo': 'c', 'Block': 'b'}
        for i, element in enumerate(self.det_result_data['compos']):
            e = Element(str(i) + class_map[element['class']], 'android', element['class'], element['position'], self.det_result_data['img_shape'])
            if element['class'] == 'Text':
                e.text_content = element['text_content']
            e.get_clip(self.img)
            self.elements.append(e)
            self.elements_mapping[e.id] = e

    def save_element_clips(self):
        clip_dir = pjoin(self.output_dir, 'clip')
        os.makedirs(clip_dir, exist_ok=True)

        for element in self.elements:
            name = pjoin(clip_dir, element.id + '.jpg')
            cv2.imwrite(name, element.clip)

    '''
    *********************
    *** Visualization ***
    *********************
    '''
    def show_detection_result(self):
        if self.det_result_imgs['merge'] is not None:
            cv2.imshow('det', cv2.resize(self.det_result_imgs['merge'], (int(self.img.shape[1] * (800 / self.img.shape[0])), 800)))
        elif self.det_result_data is not None:
            self.draw_detection_result()
            cv2.imshow('det', cv2.resize(self.det_result_imgs['merge'], (int(self.img.shape[1] * (800 / self.img.shape[0])), 800)))
        else:
            print('No detection result, run element_detection() or load_detection_result() first')
        cv2.waitKey()
        cv2.destroyAllWindows()

    def draw_detection_result(self, show_id=True):
        '''
        Draw detected elements based on det_result_data
        '''
        color_map = {'Compo': (0,255,0), 'Text': (0,0,255), 'Block':(0,255,255)}

        ratio = self.img.shape[0] / self.det_result_data['img_shape'][0]
        board = self.img.copy()
        for i, element in enumerate(self.elements):
            element.draw_element(board, ratio, color_map[element.category], show_id=show_id)
        self.det_result_imgs['merge'] = board.copy()

    def draw_popup_modal(self):
        if self.has_popup_modal:
            board_android = self.img.copy()
            for ele in self.elements:
                if ele.is_popup_modal:
                    ele.draw_element(board_android, color=(0,0,255), line=5, show_id=False)
            cv2.putText(board_android, 'popup modal', (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
            cv2.imshow('android', cv2.resize(board_android, (int(board_android.shape[1] * (800 / board_android.shape[0])), 800)))
            cv2.waitKey()
            cv2.destroyAllWindows()
