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
