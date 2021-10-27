import cv2
import json
import os
from os.path import join as pjoin
from match_compos.Compo import Compo


class GUIPair:
    def __init__(self, ui_name, input_dir='data/input', output_dir='data/output'):

        self.ui_name = ui_name
        self.input_dir = input_dir
        self.output_dir = output_dir

        # for android GUI
        self.img_path_android = pjoin(input_dir, 'A' + ui_name + '.jpg')
        self.img_android = cv2.imread(self.img_path_android)
        self.det_result_imgs_android = {'text': None, 'non-text': None, 'merge': None}  # image visualization for different stages
        self.det_result_data_android = None  # {'compos':[], 'img_shape'}
        # for ios GUI
        self.img_path_ios = pjoin(input_dir, 'I' + ui_name + '.png')
        self.img_ios = cv2.imread(self.img_path_ios)
        self.det_result_imgs_ios = {'text': None, 'non-text': None, 'merge': None}      # image visualization for different stages
        self.det_result_data_ios = None     # {'compos':[], 'img_shape'}

        self.compos_android = []    # list of Compo objects for android UI
        self.compos_ios = []        # list of Compo objects for ios UI

    def component_detection(self, is_text=True, is_nontext=True, is_merge=True):
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
            # convert compos as Compo objects
            self.cvt_compos()

    def load_detection_result(self, data_path_android=None, data_path_ios=None):
        if not data_path_android:
            data_path_android = pjoin(self.output_dir, 'merge', 'A' + self.ui_name + '.json')
        if not data_path_ios:
            data_path_ios = pjoin(self.output_dir, 'merge', 'I' + self.ui_name + '.json')
        self.det_result_data_android = json.load(open(data_path_android))
        self.det_result_data_ios = json.load(open(data_path_ios))
        # convert compos as Compo objects
        self.cvt_compos()

    def cvt_compos(self):
        '''
        Convert detection result to Compo objects
        @ det_result_data: {'compos':[], 'img_shape'}
        '''
        for i, compo in enumerate(self.det_result_data_android['compos']):
            c = Compo('a' + str(i), compo['class'], compo['position'], self.det_result_data_android['img_shape'])
            if compo['class'] == 'Text':
                c.text_content = compo['text_content']
            c.get_clip(self.img_android)
            self.compos_android.append(c)

        for i, compo in enumerate(self.det_result_data_ios['compos']):
            c = Compo('i' + str(i), compo['class'], compo['position'], self.det_result_data_ios['img_shape'])
            if compo['class'] == 'Text':
                c.text_content = compo['text_content']
            c.get_clip(self.img_ios)
            self.compos_ios.append(c)

    def show_detection_result(self):
        if self.det_result_imgs_android['merge']:
            cv2.imshow('android', cv2.resize(self.det_result_imgs_android['merge'], (int(self.img_android.shape[1] * (800 / self.img_android.shape[0])), 800)))
            cv2.imshow('ios', cv2.resize(self.det_result_imgs_ios['merge'], (int(self.img_ios.shape[1] * (800 / self.img_ios.shape[0])), 800)))
        elif self.det_result_data_android:
            self.draw_detection_result()
            cv2.imshow('android', cv2.resize(self.det_result_imgs_android['merge'], (int(self.img_android.shape[1] * (800 / self.img_android.shape[0])), 800)))
            cv2.imshow('ios', cv2.resize(self.det_result_imgs_ios['merge'], (int(self.img_ios.shape[1] * (800 / self.img_ios.shape[0])), 800)))
        else:
            print('No detection result, run component_detection() or load_detection_result() first')
        cv2.waitKey()
        cv2.destroyAllWindows()

    def draw_detection_result(self):
        '''
        Draw detected compos based on det_result_data
        '''
        color_map = {'Compo': (0,255,0), 'Text': (0,0,255)}

        def resize_bbox(bbox, resize_ratio):
            for key in bbox:
                bbox[key] = int(bbox[key] * resize_ratio)

        ratio = self.img_android.shape[0] / self.det_result_data_android['img_shape'][0]
        board = self.img_android.copy()
        for i, compo in enumerate(self.det_result_data_android['compos']):
            resize_bbox(compo['position'], ratio)
            pos = compo['position']
            cv2.rectangle(board, (pos['column_min'], pos['row_min']), (pos['column_max'], pos['row_max']), color_map[compo['class']], 2)
        self.det_result_imgs_android['merge'] = board.copy()

        ratio = self.img_ios.shape[0] / self.det_result_data_ios['img_shape'][0]
        board = self.img_ios.copy()
        for i, compo in enumerate(self.det_result_data_ios['compos']):
            resize_bbox(compo['position'], ratio)
            pos = compo['position']
            cv2.rectangle(board, (pos['column_min'], pos['row_min']), (pos['column_max'], pos['row_max']), color_map[compo['class']], 2)
        self.det_result_imgs_ios['merge'] = board.copy()
