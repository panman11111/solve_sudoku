import cv2
import numpy as np
import pytesseract
from PIL import Image


def timer(func):
    """ 実行時間を計測するデコレータ """
    import time

    def wrapper(*args, **kwargs):
        start_time = time.time()
        ret = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f'{func.__name__}: {elapsed:.4f} sec')
        return ret

    return wrapper


def is_convex(pts):
    """
    ポリゴンが凸かどうかをチェック

    Args:
        pts (list): ポリゴンの頂点座標リスト

    Returns:
        bool: ポリゴンが凸ならTrue、そうでなければFalse
    """
    pts = np.asarray(pts).reshape((-1, 2))
    n = len(pts)
    sign = 0
    for i in range(n):
        pre = (i - 1 + n) % n
        pst = (i + 1) % n
        e1 = pts[pre] - pts[i]
        e2 = pts[pst] - pts[i]
        det = e1[0] * e2[1] - e1[1] * e2[0]
        if sign == 0:
            sign = 1 if det >= 0 else -1
        else:
            if sign * det < 0:
                return False
    return True


def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """
    画像にガウシアンブラーを適用する関数

    Parameters:
    - image: 入力画像 (numpy.ndarray)
    - kernel_size: カーネルサイズ (タプル), デフォルトは (5, 5)

    Returns:
    - 処理された画像 (numpy.ndarray)
    """
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred_image


def apply_morphology_operations(image, kernel_size=3, dilation_iterations=1, erosion_iterations=1):
    """
    画像にモルフォロジー演算（膨張と収縮）を適用する関数。

    Parameters:
        image (numpy.ndarray): 入力画像
        kernel_size (int): カーネルサイズ（膨張および収縮のサイズ）
        dilation_iterations (int): 膨張の反復回数
        erosion_iterations (int): 収縮の反復回数

    Returns:
        numpy.ndarray: モルフォロジー演算を適用した結果の画像
    """
    # カーネルを作成
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 膨張を適用
    dilated_image = cv2.dilate(image, kernel, iterations=dilation_iterations)

    # 収縮を適用
    eroded_image = cv2.erode(dilated_image, kernel,
                             iterations=erosion_iterations)

    return eroded_image


@timer
def detect_edges(image, thresh1=100.0, thresh2=200.0):
    """
    Canny法を使用して一貫したエッジを検出

    Args:
        image (numpy.ndarray): 入力画像
        thresh1 (float): 低い閾値
        thresh2 (float): 高い閾値

    Returns:
        numpy.ndarray: エッジ検出結果の二値画像
    """
    # グレースケール画像に変換（カラー画像の場合）
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif image.ndim == 2:
        gray = image

    height, width = gray.shape
    struct_size = int(min(height, width) * 0.01)
    structure = np.ones((struct_size, struct_size), dtype=image.dtype)

    binary_image = cv2.Canny(gray, thresh1, thresh2)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, structure)

    return binary_image


@timer
def detect_frame(edge_image):
    """
    最大の矩形を検出

    Args:
        edge_image (numpy.ndarray): エッジ検出結果の二値画像

    Returns:
        numpy.ndarray: 最大の矩形の頂点座標
    """
    maxlen = 0.0
    maxarc = None
    contours, _ = cv2.findContours(
        edge_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        arclen = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.01 * arclen, True)

        if maxlen < arclen and len(approx) == 4 and is_convex(approx):
            maxlen = arclen
            maxarc = approx

    frame = maxarc.reshape((-1, 2))
    return frame


@timer
def extract_sudoku(gray, src_pts):
    """
    入力画像から数独領域を抽出

    Args:
        gray (numpy.ndarray): 入力画像のグレースケール版
        src_pts (numpy.ndarray): 数独領域の輪郭の頂点座標

    Returns:
        numpy.ndarray: 数独領域の画像
        numpy.ndarray: 透視変換行列
    """
    center = np.mean(src_pts, axis=1)
    pts = [(np.arctan2(p[1] - center[1], p[0] - center[0]) + np.pi, p)
           for p in src_pts]
    pts = sorted(pts, key=lambda p: p[0])
    src_pts = np.asarray([p[1] for p in pts], dtype='float32')
    dst_pts = np.asarray(
        [[0, 0], [900, 0], [900, 900], [0, 900]], dtype='float32')
    homography = cv2.getPerspectiveTransform(src_pts, dst_pts)
    sudoku_image = cv2.warpPerspective(gray, homography, (900, 900))
    return sudoku_image, homography


@timer
def image2text(sudoku_image, method='tesseract'):
    """
    数独画像から数字を抽出

    Args:
        sudoku_image (numpy.ndarray): 数独領域の画像

    Returns:
        numpy.ndarray: 数独の数字を表す行列
    """
    if method != 'tesseract':
        raise Exception(f'OCRメソッド "{method}" はサポートされていません')
    sudoku_image = cv2.adaptiveThreshold(
        sudoku_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 11)
    cells = np.zeros((9, 9), dtype='uint8')

    for i in range(9):
        for j in range(9):
            xs = j * 100 + 15
            xe = (j + 1) * 100 - 15
            ys = i * 100 + 15
            ye = (i + 1) * 100 - 15
            cell = sudoku_image[ys:ye, xs:xe]

            iy, ix = np.where(cell > 128)
            if ix.size > 0 and iy.size > 0:
                xmin = np.min(ix)
                xmax = np.max(ix)
                ymin = np.min(iy)
                ymax = np.max(iy)
                cx = (xmin + xmax) // 2
                cy = (ymin + ymax) // 2

                h, w = cell.shape
                dx = w // 2 - cx
                dy = h // 2 - cy
                M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]])
                cell = cv2.warpAffine(cell, M, (h, w))

            cell = np.pad(cell, (5, 5), mode='constant', constant_values=0)

            config = '--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789'
            cell = 255 - cell
            ratio = np.count_nonzero(cell < 128) / cell.size

            if ratio > 0.05:
                text = pytesseract.image_to_string(
                    Image.fromarray(cell), lang='eng', config=config)
                print(text)

                if text != '':
                    cells[i, j] = int(text)

    print(cells)
    return cells


def sudoku_solve_dfs(problem, start=0):
    """
    深さ優先探索による数独問題の解決

    Args:
        problem (numpy.ndarray): 数独問題の行列
        start (int): 開始位置のインデックス

    Returns:
        bool: 数独が解けた場合True、解けなかった場合False
    """
    rows, cols = problem.shape
    nums = set([i for i in range(0, 10)])

    for index in range(start, rows * cols):
        i = index // cols
        j = index % cols
        if problem[i, j] == 0:
            k = i // 3
            l = j // 3
            row_nums = problem[i, :].tolist()
            col_nums = problem[:, j].tolist()
            blk_nums = problem[3*k:3*k+3, 3*l:3*l+3].reshape(-1).tolist()
            used = set(row_nums + col_nums + blk_nums)
            diff = nums.difference(used)
            if len(diff) == 0:
                return False

            success = False
            for n in diff:
                problem[i, j] = n
                if sudoku_solve_dfs(problem, i * cols + j):
                    success = True
                    break
                problem[i, j] = 0

            if not success:
                return False

    return True


@timer
def sudoku_solve(problem):
    """
    数独問題の解決

    Args:
        problem (numpy.ndarray): 数独問題の行列

    Returns:
        numpy.ndarray or None: 解が存在する場合は解答行列、存在しない場合はNone
    """
    answer = problem.copy()
    if sudoku_solve_dfs(answer):
        return answer
    return None


def main(filename):
    """
    数独画像を解析して解答を生成

    Args:
        filename (str): 入力画像ファイル名

    Returns:
        numpy.ndarray: 解答を含む画像
    """
    # 入力画像をロード
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    if image is None:
        raise Exception('画像ファイルの読み込みに失敗しました: ' + filename)

    # 画像が大きすぎる場合、リサイズ
    height, width, _ = image.shape
    if max(height, width) >= 1500:
        scale = 1500 / max(height, width)
        image = cv2.resize(image, None, fx=scale, fy=scale)
        height, width, _ = image.shape

    # カラーフォーマットに変換
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # ガウシアンフィルターを適用
    blurred_image = apply_gaussian_blur(gray)

    # モルフォロジー演算を適用
    morphology_photo = apply_morphology_operations(blurred_image)

    # エッジ検出を行う
    edges = detect_edges(morphology_photo)

    # 最大の矩形を検出
    frame_pts = detect_frame(edges)

    # 数独問題を検出
    sudoku_image, homography = extract_sudoku(gray, frame_pts)
    problem = image2text(sudoku_image, method='tesseract')

    # 数独を解く
    answer = sudoku_solve(problem)
    if answer is None:
        raise Exception('数独問題の解決に失敗しました！')

    # 解答を元の画像に埋め込む
    overlay = np.zeros((900, 900, 4), dtype='float32')
    font = cv2.FONT_HERSHEY_PLAIN

    for i in range(9):
        for j in range(9):
            if problem[i, j] == 0:
                x = j * 100 + 20
                y = i * 100 + 100 - 20
                cv2.putText(overlay, str(
                    answer[i, j]), (x, y), font, 5.0, (1, 0, 0, 1), 5, cv2.LINE_AA)

    warp_numbers = cv2.warpPerspective(
        overlay, np.linalg.inv(homography), (width, height))
    alpha = warp_numbers[:, :, 3:4]
    rgb = warp_numbers[:, :, :3]
    result = (1.0 - alpha) * (image / 255.0).astype('float32') + alpha * rgb

    return result
