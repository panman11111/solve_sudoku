import streamlit as st
from sudoku_solve import main
import tempfile

st.title('数独アプリ')

# 画像アップロード
uploaded_image = st.file_uploader("数独の画像をアップロードしてください", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    st.image(uploaded_image, caption='アップロードされた画像', use_column_width=True)

    if st.button("解答を表示"):
        # 画像を一時ファイルとして保存
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_image.read())
            tmp_path = tmp_file.name
        
        # 数独を解析して解答を生成
        result_image = main(tmp_path)

        # 解いた数独を表示
        st.image(result_image, caption='解かれた数独', use_column_width=True)
