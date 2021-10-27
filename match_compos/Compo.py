import cv2


class Compo:
    def __init__(self, compo_id, category, position, img_size, text_content=None):
        self.id = compo_id
        self.category = category
        self.col_min, self.row_min, self.col_max, self.row_max = int(position['column_min']), int(position['row_min']), int(position['column_max']), int(position['row_max'])
        self.width = self.col_max - self.col_min
        self.height = self.row_max - self.row_min
        self.area = self.width * self.height

        self.img_size = img_size    # the size of the resized image while detection
        self.clip = None

        self.text_content = text_content

    def init_bound(self):
        self.width = self.col_max - self.col_min
        self.height = self.row_max - self.row_min
        self.area = self.width * self.height

    def get_clip(self, org_img):
        ratio = org_img.shape[0] / self.img_size[0]
        left, right, top, bottom = int(self.col_min * ratio), int(self.col_max * ratio), int(self.row_min * ratio), int(self.row_max * ratio)
        self.clip = org_img[top: bottom, left: right]

    def resize_bound(self, resize_ratio):
        return [int(self.col_min*resize_ratio), int(self.row_min*resize_ratio), int(self.col_max*resize_ratio), int(self.row_max*resize_ratio)]

    def draw_compo(self, board, ratio=None, color=(0,255,0), line=2, show_id=True):
        if not ratio:
            ratio = board.shape[0] / self.img_size
        bound = self.resize_bound(ratio)
        cv2.rectangle(board, (bound[0], bound[1]), (bound[2], bound[3]), color, line)
        if show_id:
            cv2.putText(board, self.id, (bound[0] + 3, bound[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
