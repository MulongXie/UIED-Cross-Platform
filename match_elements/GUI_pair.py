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


class GUIPair:
    def __init__(self, ui_name, input_dir='data/input', output_dir='data/output', detection_resize_height=900,
                 img_path_android=None, img_path_ios=None):
        self.ui_name = ui_name
        self.input_dir = input_dir
        self.output_dir = output_dir

        self.detection_resize_height = detection_resize_height  # resize the input gui while detecting
        # for android GUI
        self.img_path_android = pjoin(input_dir, 'A' + ui_name + '.jpg') if not img_path_android else img_path_android
        self.img_android = cv2.imread(self.img_path_android)
        self.det_result_imgs_android = {'text': None, 'non-text': None, 'merge': None}  # image visualization for different stages
        self.det_result_data_android = None  # {'compos':[], 'img_shape'}
        # for ios GUI
        self.img_path_ios = pjoin(input_dir, 'I' + ui_name + '.png') if not img_path_ios else img_path_ios
        self.img_ios = cv2.imread(self.img_path_ios)
        self.det_result_imgs_ios = {'text': None, 'non-text': None, 'merge': None}      # image visualization for different stages
        self.det_result_data_ios = None     # {'compos':[], 'img_shape'}

        self.elements_android = []          # list of Element objects for android UI
        self.elements_ios = []              # list of Element objects for ios UI
        self.elements_mapping = {}          # {'id': Element}
        self.element_matching_pairs = []    # list of matching similar element pairs: [(ele_android, ele_ios)]

        self.has_popup_modal_android = False        # if the ui has popup modal
        self.has_popup_modal_ios = False            # if the ui has popup modal

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
        img_android_resized = cv2.resize(self.img_android, (int(self.img_android.shape[1] * (self.detection_resize_height / self.img_android.shape[0])), self.detection_resize_height))
        img_ios_resized = cv2.resize(self.img_ios, (int(self.img_ios.shape[1] * (self.detection_resize_height / self.img_ios.shape[0])), self.detection_resize_height))
        area_resize_a = img_android_resized.shape[0] * img_android_resized.shape[1]
        area_resize_i = img_ios_resized.shape[0] * img_ios_resized.shape[1]

        sum_gray_a = np.sum(cv2.cvtColor(img_android_resized, cv2.COLOR_BGR2GRAY))
        sum_gray_i = np.sum(cv2.cvtColor(img_ios_resized, cv2.COLOR_BGR2GRAY))

        if sum_gray_a / (img_android_resized.shape[0] * img_android_resized.shape[1]) < 100:
            for ele in self.elements_android:
                if ele.category == 'Compo' and \
                        ele.height / ele.detection_img_size[0] > height_thresh and ele.width / ele.detection_img_size[1] > width_thresh:
                    ele.get_clip(img_android_resized)
                    if is_element_modal(ele, area_resize_a):
                        self.has_popup_modal_android = True
                        ele.is_popup_modal = True
        if sum_gray_i / (img_ios_resized.shape[0] * img_ios_resized.shape[1]) < 100:
            for ele in self.elements_ios:
                if ele.category == 'Compo' and \
                        ele.height / ele.detection_img_size[0] > height_thresh and ele.width / ele.detection_img_size[1] > width_thresh:
                    ele.get_clip(img_ios_resized)
                    if is_element_modal(ele, area_resize_i):
                        self.has_popup_modal_ios = True
                        ele.is_popup_modal = True
        if not self.has_popup_modal_android and not self.has_popup_modal_ios:
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
            self.det_result_imgs_android['text'] = text.text_detection_paddle(self.img_path_android, pjoin(self.output_dir, 'ocr'), paddle_cor=paddle_cor)
            self.det_result_imgs_ios['text'] = text.text_detection_paddle(self.img_path_ios, pjoin(self.output_dir, 'ocr'), paddle_cor=paddle_cor)
        if is_nontext:
            os.makedirs(pjoin(self.output_dir, 'ip'), exist_ok=True)
            import detect_compo.ip_region_proposal as ip
            key_params = {'min-grad': 6, 'ffl-block': 5, 'min-ele-area': 100, 'merge-contained-ele': True}
            self.det_result_imgs_android['non-text'] = ip.compo_detection(self.img_path_android, self.output_dir, key_params, resize_by_height=self.detection_resize_height, adaptive_binarization=False)
            self.det_result_imgs_ios['non-text'] = ip.compo_detection(self.img_path_ios, self.output_dir, key_params, resize_by_height=self.detection_resize_height, adaptive_binarization=False)
        if is_merge:
            os.makedirs(pjoin(self.output_dir, 'merge'), exist_ok=True)
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

    def save_element_clips(self):
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

    '''
    ******************************
    *** Match Similar Elements ***
    ******************************
    '''
    def match_similar_elements(self, min_similarity_img=0.75, min_similarity_text=0.8, img_sim_method='dhash', del_prev=True):
        '''
        @min_similarity_img: similarity threshold for Non-text elements
        @min_similarity_text: similarity threshold for Text elements
        @img_sim_method: the method used to calculate the similarity between two images
            options: 'dhash', 'ssim', 'sift', 'surf'
        @del_prev: if to delete all previously matched compos
        '''
        if del_prev:
            self.element_matching_pairs = []
            for ele in self.elements_ios + self.elements_android:
                ele.matched_element = None

        start = time.clock()
        if img_sim_method == 'resnet':
            from keras.applications.resnet50 import ResNet50
            resnet_model = ResNet50(include_top=False, input_shape=(32, 32, 3))
        else:
            resnet_model = None

        mark = np.full(len(self.elements_ios), False)
        n_compos = 0
        n_texts = 0
        for ele_a in self.elements_android:
            for j, ele_b in enumerate(self.elements_ios):
                # only match elements in the same category
                if ele_b.matched_element is not None or ele_a.category != ele_b.category:
                    continue
                if mark[j] or \
                        max(ele_a.height, ele_b.height) / min(ele_a.height, ele_b.height) > 2 or max(ele_a.width, ele_b.width) / min(ele_a.width, ele_b.width) > 2:
                    continue
                # use different method to calc the similarity of of images and texts
                if ele_a.category == 'Compo':
                    # match non-text clip through image similarity
                    compo_similarity = match.image_similarity(ele_a.clip, ele_b.clip, method=img_sim_method, resnet_model=resnet_model)
                    if compo_similarity > min_similarity_img:
                        n_compos += 1
                        self.element_matching_pairs.append((ele_a, ele_b))
                        ele_a.matched_element = ele_b
                        ele_b.matched_element = ele_a
                        mark[j] = True
                        break
                elif ele_a.category == 'Text':
                    # match text by through string similarity
                    text_similarity = match.text_similarity(ele_a.text_content, ele_b.text_content)
                    if text_similarity > min_similarity_text:
                        n_texts += 1
                        self.element_matching_pairs.append((ele_a, ele_b))
                        ele_a.matched_element = ele_b
                        ele_b.matched_element = ele_a
                        mark[j] = True
                        break
        print('[Similar Elements Matching %.3fs] Method:%s Paired Text:%d, Paired Compos:%d' % ((time.clock() - start), img_sim_method, n_compos, n_texts))

    def save_matched_element_pairs_clips(self, category='Compo', start_file_id=None, rm_exit=False, output_dir='data/output/matched_compos'):
        '''
        Save the clips of matched element pairs
        @category: "Compo" or "Text"
        @start_file_id: where the saved clip file name start with
        @rm_exit: if remove all previously saved clips
        @output_dir: the root directory for saving
        '''
        if len(self.element_matching_pairs) == 0:
            print('No similar compos matched, run match_similar_elements first')
            return
        if rm_exit:
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        if start_file_id is None:
            files = glob(pjoin(output_dir, '*'))
            file_ids = [int(f.replace('\\', '/').split('/')[-1].split('_')[0]) for f in files]
            start_file_id = max(file_ids) if len(file_ids) > 0 else 0

        for pair in self.element_matching_pairs:
            if pair[0].category == category:
                cv2.imwrite(pjoin(output_dir, str(start_file_id) + '_a.jpg'), pair[0].clip)
                cv2.imwrite(pjoin(output_dir, str(start_file_id) + '_i.jpg'), pair[1].clip)
            start_file_id += 1
        print('Save matched compo pairs to', output_dir)

    '''
    *********************
    *** Visualization ***
    *********************
    '''
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
        color_map = {'Compo': (0,255,0), 'Text': (0,0,255), 'Block':(0,255,255)}

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

    def visualize_matched_element_pairs(self, line=-1):
        board_android = self.img_android.copy()
        board_ios = self.img_ios.copy()
        for pair in self.element_matching_pairs:
            color = (rint(0,255), rint(0,255), rint(0,255))
            pair[0].draw_element(board_android, color=color, line=line, show_id=False)
            pair[1].draw_element(board_ios, color=color, line=line, show_id=False)
        cv2.imshow('android', cv2.resize(board_android, (int(board_android.shape[1] * (800 / board_android.shape[0])), 800)))
        cv2.imshow('ios', cv2.resize(board_ios, (int(board_ios.shape[1] * (800 / board_ios.shape[0])), 800)))
        cv2.waitKey()
        cv2.destroyAllWindows()

    def draw_popup_modal(self):
        if self.has_popup_modal_android:
            board_android = self.img_android.copy()
            for ele in self.elements_android:
                if ele.is_popup_modal:
                    ele.draw_element(board_android, color=(0,0,255), line=5, show_id=False)
            cv2.putText(board_android, 'popup modal', (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
            cv2.imshow('android', cv2.resize(board_android, (int(board_android.shape[1] * (800 / board_android.shape[0])), 800)))
        if self.has_popup_modal_ios:
            board_ios = self.img_ios.copy()
            for ele in self.elements_ios:
                if ele.is_popup_modal:
                    ele.draw_element(board_ios, color=(0,0,255), line=5, show_id=False)
            cv2.putText(board_ios, 'popup modal', (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.imshow('ios', cv2.resize(board_ios, (int(board_ios.shape[1] * (800 / board_ios.shape[0])), 800)))
        cv2.waitKey()
        cv2.destroyAllWindows()
