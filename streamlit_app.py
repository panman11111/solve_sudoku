import streamlit as st
# import cv2
import numpy as np
from PIL import Image
from sudoku_solve import main
# import tempfile

st.title('数独アプリ')

# 画像アップロード
uploaded_image = st.file_uploader("数独の画像をアップロードしてください", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    st.image(uploaded_image, caption='アップロードされた画像', use_column_width=True)

    if st.button("解答を表示"):
        image = np.array(Image.open(uploaded_image))

        # OpenCVで処理した画像を一時的なファイルに保存
        # with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image:
        #     temp_image_path = temp_image.name
        #     cv2.imwrite(temp_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        result_image = main(image)

        # 解いた数独を表示
        st.image(result_image, caption='解かれた数独', use_column_width=True)
