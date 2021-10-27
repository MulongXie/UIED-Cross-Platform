class Compo:
    def __init__(self, compo_id, category, position, img_size, text_content=None):
        self.id = compo_id
        self.category = category
        self.col_min, self.row_min, self.col_max, self.row_max = position
        self.width = self.col_max - self.col_min
        self.height = self.row_max - self.row_min
        self.area = self.width * self.height

        self.img_size = img_size
        self.clip = None

        self.text_content = text_content

    def init_bound(self):
        self.width = self.col_max - self.col_min
        self.height = self.row_max - self.row_min
        self.area = self.width * self.height

    def clip(self, org_img):
        ratio = org_img.shape[0] / self.img_size[0]
        left, right, top, bottom = self.col_min * ratio, self.col_max * ratio, self.row_min * ratio, self.row_max * ratio
        self.clip = org_img[top: bottom, left: right]
