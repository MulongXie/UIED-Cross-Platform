import cv2


class Element:
    def __init__(self, element_id, ui_type, category, position, img_size, text_content=None):
        self.id = element_id
        self.category = category        # Compo / Text
        self.text_content = text_content

        self.col_min, self.row_min, self.col_max, self.row_max = int(position['column_min']), int(position['row_min']), int(position['column_max']), int(position['row_max'])
        self.width = self.col_max - self.col_min
        self.height = self.row_max - self.row_min
        self.area = self.width * self.height

        self.detection_img_size = img_size    # the size of the resized image while detection
        self.clip = None

        self.ui_type = ui_type          # android/ios
        self.matched_element = None     # the matched Element in another ui
        self.is_popup_modal = False     # if the element is popup modal

    def init_bound(self):
        self.width = self.col_max - self.col_min
        self.height = self.row_max - self.row_min
        self.area = self.width * self.height

    def get_clip(self, org_img):
        ratio = org_img.shape[0] / self.detection_img_size[0]
        left, right, top, bottom = int(self.col_min * ratio), int(self.col_max * ratio), int(self.row_min * ratio), int(self.row_max * ratio)
        self.clip = org_img[top: bottom, left: right]

    def resize_bound(self, resize_ratio):
        return [int(self.col_min*resize_ratio), int(self.row_min*resize_ratio), int(self.col_max*resize_ratio), int(self.row_max*resize_ratio)]

    def draw_element(self, board, ratio=None, color=(0,255,0), line=2, show_id=True, show=False):
        if not ratio:
            ratio = board.shape[0] / self.detection_img_size[0]
        bound = self.resize_bound(ratio)
        cv2.rectangle(board, (bound[0], bound[1]), (bound[2], bound[3]), color, line)
        if show_id:
            cv2.putText(board, self.id, (bound[0] + 3, bound[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        if show:
            cv2.imshow(self.id, cv2.resize(board, (int(board.shape[1] * (800 / board.shape[0])), 800)))
            cv2.waitKey()
            cv2.destroyWindow(self.id)
