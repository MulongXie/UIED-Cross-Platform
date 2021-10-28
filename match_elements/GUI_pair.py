import cv2
import json
import os
from os.path import join as pjoin
from match_elements.Element import Element
import match_elements.matching as match


class GUIPair:
    def __init__(self, ui_name, input_dir='data/input', output_dir='data/output'):

        self.ui_name = ui_name
        self.input_dir = input_dir
        self.output_dir = output_dir

        # for android GUI
        self.img_path_android = pjoin(input_dir, 'A' + ui_name + '.jpg')
        self.img_android = cv2.imread(self.img_path_android)
        self.det_result_imgs_android = {'text': None, 'non-text': None, 'merge': None}  # image visualization for different stages
        self.det_result_data_android = None  # {'elements':[], 'img_shape'}
        # for ios GUI
        self.img_path_ios = pjoin(input_dir, 'I' + ui_name + '.png')
        self.img_ios = cv2.imread(self.img_path_ios)
        self.det_result_imgs_ios = {'text': None, 'non-text': None, 'merge': None}      # image visualization for different stages
        self.det_result_data_ios = None     # {'elements':[], 'img_shape'}

        self.elements_android = []    # list of Element objects for android UI
        self.elements_ios = []        # list of Element objects for ios UI
        self.elements_mapping = {}    # {'id': Element}

    def element_detection(self, is_text=True, is_nontext=True, is_merge=True):
        if is_text:
            import detect_text.text_detection as text
            from paddleocr import PaddleOCR
            paddle_cor = PaddleOCR(use_angle_cls=True, lang="ch")
            self.det_result_imgs_android['text'] = text.text_detection_paddle(self.img_path_android, self.output_dir, paddle_cor=paddle_cor)
            self.det_result_imgs_ios['text'] = text.text_detection_paddle(self.img_path_ios, self.output_dir, paddle_cor=paddle_cor)
        if is_nontext:
            import detect_compo.ip_region_proposal as ip
            key_params = {'min-grad': 6, 'ffl-block': 5, 'min-ele-area': 50, 'merge-contained-ele': True, 'resize_by_height': 900}
            self.det_result_imgs_android['non-text'] = ip.compo_detection(self.img_path_android, self.output_dir, key_params)
            self.det_result_imgs_ios['non-text'] = ip.compo_detection(self.img_path_ios, self.output_dir, key_params)
        if is_merge:
            import detect_merge.merge as merge
            # for android GUI
            compo_path = pjoin(self.output_dir, 'ip', 'A' + str(self.ui_name) + '.json')
            ocr_path = pjoin(self.output_dir, 'ocr', 'A' + str(self.ui_name) + '.json')
            self.det_result_imgs_android['merge'], self.det_result_data_android = merge.merge(self.img_path_android, compo_path, ocr_path, pjoin(self.output_dir, 'merge'), is_remove_bar=True, is_paragraph=False)
            # for ios GUI
            compo_path = pjoin(self.output_dir, 'ip', 'I' + str(self.ui_name) + '.json')
            ocr_path = pjoin(self.output_dir, 'ocr', 'I' + str(self.ui_name) + '.json')
            self.det_result_imgs_ios['merge'], self.det_result_data_ios = merge.merge(self.img_path_ios, compo_path, ocr_path, pjoin(self.output_dir, 'merge'), is_remove_bar=True, is_paragraph=False)
            # convert elements as Element objects
            self.cvt_elements()

    def load_detection_result(self, data_path_android=None, data_path_ios=None):
        if not data_path_android:
            data_path_android = pjoin(self.output_dir, 'merge', 'A' + self.ui_name + '.json')
        if not data_path_ios:
            data_path_ios = pjoin(self.output_dir, 'merge', 'I' + self.ui_name + '.json')
        self.det_result_data_android = json.load(open(data_path_android))
        self.det_result_data_ios = json.load(open(data_path_ios))
        # convert elements as Element objects
        self.cvt_elements()

    def cvt_elements(self):
        '''
        Convert detection result to Element objects
        @ det_result_data: {'elements':[], 'img_shape'}
        '''
        class_map = {'Text': 't', 'Compo': 'c'}
        for i, element in enumerate(self.det_result_data_android['compos']):
            e = Element('a' + str(i) + class_map[element['class']], 'android', element['class'], element['position'], self.det_result_data_android['img_shape'])
            if element['class'] == 'Text':
                e.text_content = element['text_content']
            e.get_clip(self.img_android)
            self.elements_android.append(e)
            self.elements_mapping[e.id] = e

        for i, element in enumerate(self.det_result_data_ios['compos']):
            e = Element('i' + str(i) + class_map[element['class']], 'ios', element['class'], element['position'], self.det_result_data_ios['img_shape'])
            if element['class'] == 'Text':
                e.text_content = element['text_content']
            e.get_clip(self.img_ios)
            self.elements_ios.append(e)
            self.elements_mapping[e.id] = e

    def show_detection_result(self):
        if self.det_result_imgs_android['merge'] is not None:
            cv2.imshow('android', cv2.resize(self.det_result_imgs_android['merge'], (int(self.img_android.shape[1] * (800 / self.img_android.shape[0])), 800)))
            cv2.imshow('ios', cv2.resize(self.det_result_imgs_ios['merge'], (int(self.img_ios.shape[1] * (800 / self.img_ios.shape[0])), 800)))
        elif self.det_result_data_android is not None:
            self.draw_detection_result()
            cv2.imshow('android', cv2.resize(self.det_result_imgs_android['merge'], (int(self.img_android.shape[1] * (800 / self.img_android.shape[0])), 800)))
            cv2.imshow('ios', cv2.resize(self.det_result_imgs_ios['merge'], (int(self.img_ios.shape[1] * (800 / self.img_ios.shape[0])), 800)))
        else:
            print('No detection result, run element_detection() or load_detection_result() first')
        cv2.waitKey()
        cv2.destroyAllWindows()

    def draw_detection_result(self, show_id=True):
        '''
        Draw detected elements based on det_result_data
        '''
        color_map = {'Compo': (0,255,0), 'Text': (0,0,255)}

        ratio = self.img_android.shape[0] / self.det_result_data_android['img_shape'][0]
        board = self.img_android.copy()
        for i, element in enumerate(self.elements_android):
            element.draw_element(board, ratio, color_map[element.category], show_id=show_id)
        self.det_result_imgs_android['merge'] = board.copy()

        ratio = self.img_ios.shape[0] / self.det_result_data_ios['img_shape'][0]
        board = self.img_ios.copy()
        for i, element in enumerate(self.elements_ios):
            element.draw_element(board, ratio, color_map[element.category], show_id=show_id)
        self.det_result_imgs_ios['merge'] = board.copy()

    def save_clips(self):
        clip_dir = pjoin(self.output_dir, 'clip')
        clip_dir_android = pjoin(clip_dir, 'android')
        clip_dir_ios = pjoin(clip_dir, 'ios')
        os.makedirs(clip_dir, exist_ok=True)
        os.makedirs(clip_dir_android, exist_ok=True)
        os.makedirs(clip_dir_ios, exist_ok=True)

        for element in self.elements_android:
            name = pjoin(clip_dir_android, element.id + '.jpg')
            cv2.imwrite(name, element.clip)
        for element in self.elements_ios:
            name = pjoin(clip_dir_ios, element.id + '.jpg')
            cv2.imwrite(name, element.clip)
