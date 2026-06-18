# SVM Kernel Trick 3D Interactive Demo 開發規格書

> Version: v1.0  
> Output Format: Markdown  
> Target Implementation Agent: Antigravity Agent / Coding Agent  
> Language: zh-TW 說明 + English code comments  
> Project Type: Machine Learning Teaching Demo / Manim Animation / Interactive Web App

---

## 1. 專案總覽

### 1.1 專案名稱

**SVM Kernel Trick 3D Interactive Demo**

### 1.2 專案目標

建立一個完整的 **Support Vector Machine Kernel Trick 教學展示系統**，讓學生可以透過動畫、真實模型視覺化與互動式網頁，理解：

1. 為什麼某些資料在 2D 平面中無法用直線分開。
2. 如何透過特徵映射將資料提升到較高維度。
3. 為什麼在高維 feature space 中可以使用 hyperplane 分類。
4. 3D hyperplane 投影回 2D 後，會形成非線性 decision boundary。
5. 真正的 RBF kernel SVM 並不是單純映射到 3D，而是對應到高維或無限維特徵空間。
6. 如何使用 Streamlit + Plotly 讓學生調整 `C`、`gamma`、`kernel` 等參數，觀察 SVM 決策邊界變化。

### 1.3 目標使用者

- 高中資訊課學生
- 大學初階機器學習學生
- 機器學習入門自學者
- 教師課堂展示
- NotebookLM / Manim / Streamlit 教學素材製作者

### 1.4 核心教學故事

本專案使用「中心藍點、外圈紅點」的資料分布作為主題：

1. 在原始 2D 平面中，中心藍點與外圈紅點無法用一條直線分開。
2. 透過 feature mapping 將資料從 2D 提升到 3D。
3. 使用簡化的教學映射：

   $$
   \phi(x, y) = (x, y, x^2 + y^2)
   $$

4. 在 3D feature space 中，內圈藍點位於低處，外圈紅點位於高處。
5. 此時可以用一個水平 hyperplane 分開兩群資料。
6. 這個 3D hyperplane 投影回 2D，形成一個圓形 decision boundary。
7. 接著使用真正的 sklearn RBF SVM，展示實際 decision function surface。
8. 最後提供互動式 Streamlit App，讓學生調整參數並觀察模型變化。

---

## 2. 技術架構

### 2.1 技術棧

| 模組 | 技術 |
|---|---|
| 概念動畫 | Python, Manim Community Edition, NumPy |
| 機器學習 | scikit-learn, NumPy, matplotlib |
| 互動展示 | Streamlit, Plotly, pandas, scikit-learn |
| 資料產生 | NumPy |
| 視覺化 | Manim, matplotlib 3D, Plotly Contour, Plotly Surface |

### 2.2 預期 Repository 結構

```text
svm-kernel-trick-3d-demo/
├── README.md
├── requirements.txt
├── phase1_manim_kernel_trick.py
├── phase2_rbf_decision_surface.py
├── phase3_streamlit_app.py
├── utils/
│   ├── __init__.py
│   ├── data_generator.py
│   └── svm_utils.py
├── assets/
└── outputs/
```

### 2.3 requirements.txt

```txt
manim
numpy
scikit-learn
matplotlib
streamlit
plotly
pandas
```

### 2.4 安裝與執行指令

```bash
pip install -r requirements.txt
```

Manim 預覽：

```bash
manim -pql phase1_manim_kernel_trick.py SVMKernelTrick3D
```

Manim 高畫質輸出：

```bash
manim -pqh phase1_manim_kernel_trick.py SVMKernelTrick3D
```

Phase 2 RBF SVM 決策曲面：

```bash
python phase2_rbf_decision_surface.py
```

Phase 3 Streamlit App：

```bash
streamlit run phase3_streamlit_app.py
```

---

## 3. Phase 1：Manim Concept Animation

### 3.1 Phase 目標

建立一個乾淨、清楚、適合教學的 Manim 3D 動畫，展示 SVM Kernel Trick 的直觀概念。

本 Phase 使用簡化的特徵映射：

$$
z = x^2 + y^2
$$

讓學生理解：

- 2D 中非線性不可分
- 提升到 3D 後可以線性可分
- 3D hyperplane 投影回 2D 會形成非線性邊界

### 3.2 輸出檔案

```text
phase1_manim_kernel_trick.py
```

### 3.3 Scene 設定

| 項目 | 規格 |
|---|---|
| Scene Class | `SVMKernelTrick3D` |
| Base Class | `ThreeDScene` |
| 主要物件 | `ThreeDAxes`, `Dot3D`, `Surface`, `ParametricFunction`, `MathTex`, `Text`, `VGroup` |
| 背景 | Dark background |
| 內圈點 | Blue |
| 外圈點 | Red |
| Hyperplane | Yellow, translucent |
| Paraboloid Surface | Translucent |
| Decision Boundary | Yellow circle |

### 3.4 資料設計

固定使用 ring dataset。

| 類別 | 說明 | Label | 顏色 | 半徑範圍 | 點數 |
|---|---|---:|---|---|---:|
| Blue Class | 靠近中心的資料點 | 0 | Blue | 0.0 ~ 1.0 | 35 |
| Red Class | 外圈環狀資料點 | 1 | Red | 1.6 ~ 2.5 | 45 |

Random Seed：

```python
random_seed = 7
```

### 3.5 動畫流程

#### Step 1：Opening Title

顯示標題：

```text
SVM Kernel Trick: From 2D to 3D
```

副標題：

```text
Nonlinear in 2D, linear in feature space.
```

#### Step 2：Show 2D Data

在 `z = 0` 平面顯示：

- 中心藍點
- 外圈紅點

文字提示：

```text
No straight line can separate them in 2D.
```

#### Step 3：Show Mapping Formula

顯示公式：

$$
\phi(x, y) = (x, y, x^2 + y^2)
$$

視覺說明每個點會依照距離原點的遠近被抬升。

#### Step 4：Animate Lifting to 3D

將每個點從：

$$
(x, y, 0)
$$

轉換為：

$$
(x, y, x^2 + y^2)
$$

預期效果：

- 藍點因為靠近中心，所以高度較低。
- 紅點因為距離中心較遠，所以高度較高。

#### Step 5：Show Paraboloid Surface

繪製半透明曲面：

$$
z = x^2 + y^2
$$

資料點需保持可見。

#### Step 6：Show Separating Hyperplane

繪製水平分割平面：

$$
z = c
$$

其中 `c` 位於藍點高度與紅點高度之間。

標籤：

```text
Hyperplane in feature space
```

#### Step 7：Project Back to 2D

說明：

$$
z = c
$$

且：

$$
z = x^2 + y^2
$$

因此：

$$
x^2 + y^2 = c
$$

在 2D 平面繪製圓形 decision boundary。

#### Step 8：Camera Rotation

緩慢旋轉 3D Camera，讓學生看懂空間分離效果。

建議 Camera 設定：

```python
initial_phi_degrees = 65
initial_theta_degrees = -45
ambient_rotation_rate = 0.18
```

#### Step 9：Final Summary

顯示總結：

```text
In 3D: linear hyperplane
In 2D: nonlinear decision boundary
This is the intuition behind the kernel trick.
```

### 3.6 Phase 1 成功標準

- 能清楚展示 2D 中無法線性分開。
- 點會依照 `z = x^2 + y^2` 提升到 3D。
- 3D hyperplane 能明顯分開兩類資料。
- 2D 投影後可看到圓形 decision boundary。
- 公式清楚可讀，動畫節奏適合教學。

---

## 4. Phase 2：Real RBF SVM Decision Function Surface

### 4.1 Phase 目標

實作真正的 sklearn `SVC(kernel="rbf")` 模型，並視覺化：

1. 原始 2D 資料點
2. 2D decision boundary
3. margin contours
4. support vectors
5. 3D decision function surface

此 Phase 需特別強調：

> Phase 1 的 `z = x^2 + y^2` 是教學用簡化映射。真正的 RBF kernel 並不是單純映射到 3D，而是隱式映射到高維或無限維特徵空間。

### 4.2 輸出檔案

```text
phase2_rbf_decision_surface.py
```

### 4.3 核心公式

RBF Kernel：

$$
K(x, x') = \exp(-\gamma ||x - x'||^2)
$$

SVM Decision Function：

$$
f(x) = \sum_i \alpha_i y_i K(x_i, x) + b
$$

Decision Boundary：

$$
f(x, y) = 0
$$

### 4.4 資料設計

使用與 Phase 1 相同的 ring dataset。

建議加入 noise：

```python
default_noise = 0.08
```

目的：

- 讓資料更接近真實狀況。
- 讓 decision boundary 與 support vectors 更有教學價值。

### 4.5 模型設定

使用 sklearn：

```python
from sklearn.svm import SVC

clf = SVC(
    kernel="rbf",
    C=10,
    gamma=1
)
```

必要方法：

```python
clf.fit(X, y)
clf.decision_function(grid_points)
clf.predict(X)
```

### 4.6 視覺化要求

#### 4.6.1 2D Dataset with Decision Boundary

需顯示：

- 原始 2D 資料點
- `f(x, y) = 0` decision boundary
- `f(x, y) = -1` margin contour
- `f(x, y) = +1` margin contour
- Support vectors

#### 4.6.2 3D Decision Function Surface

需顯示：

- 3D 曲面：

  $$
  z = f(x, y)
  $$

- 曲面高度代表模型信心。
- `z = 0` 作為分類閾值。
- 支援從不同角度觀察曲面。
- 可使用 matplotlib 3D 作為主要版本。
- 可選擇 Plotly 作為進階版本。

#### 4.6.3 Support Vectors

Support vectors 需在 2D 與 3D 圖中明顯標記。

建議樣式：

- 外圈 marker
- 比一般點更大
- 不改變 class color，避免混淆類別

### 4.7 Utility Functions

#### `utils/data_generator.py`

```python
def generate_ring_dataset(
    n_inner: int,
    n_outer: int,
    inner_radius_range: tuple[float, float],
    outer_radius_range: tuple[float, float],
    noise: float,
    random_seed: int
):
    """
    Generate a 2D ring dataset for SVM kernel trick demonstration.
    Returns:
        X: shape (n_samples, 2)
        y: shape (n_samples,)
    """
```

#### `utils/svm_utils.py`

```python
def train_svm(
    X,
    y,
    kernel: str = "rbf",
    C: float = 10.0,
    gamma: float | str = 1.0,
    degree: int = 3
):
    """
    Train sklearn SVC model.
    """
```

```python
def make_decision_grid(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    resolution: int
):
    """
    Create mesh grid for decision function visualization.
    """
```

```python
def compute_decision_surface(
    model,
    grid_points,
    xx,
    yy
):
    """
    Compute decision_function values and reshape them to mesh grid.
    """
```

### 4.8 參數教學行為

| 參數狀況 | 預期行為 |
|---|---|
| `gamma` 小 | 邊界較平滑，單一資料點影響範圍較大 |
| `gamma` 大 | 邊界更彈性，但可能 overfit |
| `C` 小 | soft margin，容許更多分類錯誤 |
| `C` 大 | 更努力分類正確，可能導致邊界變複雜 |

### 4.9 Phase 2 成功標準

- 成功訓練 sklearn RBF SVM。
- 2D decision boundary 清楚可見。
- 3D decision function surface 清楚可見。
- Support vectors 有被標記。
- 程式碼清楚區分「教學映射」與「真正 RBF SVM」。

---

## 5. Phase 3：Interactive Streamlit and Plotly Demo

### 5.1 Phase 目標

建立互動式 Web App，讓學生可以調整 SVM 參數並即時觀察：

- Decision boundary
- Margin
- Support vectors
- 3D decision function surface
- Overfitting / underfitting 趨勢

### 5.2 輸出檔案

```text
phase3_streamlit_app.py
```

### 5.3 App 標題

```text
Interactive SVM Kernel Trick 3D Demo
```

### 5.4 Sidebar Controls

| 控制項 | 類型 | 預設值 | 說明 |
|---|---|---|---|
| kernel | selectbox | rbf | linear / poly / rbf / sigmoid |
| C | slider | 10.0 | 控制 margin 與錯誤容忍度 |
| gamma | slider | 1.0 | 控制 RBF 影響範圍 |
| degree | slider | 3 | poly kernel 使用 |
| noise | slider | 0.08 | 控制資料雜訊 |
| number_of_points | slider | 120 | 控制資料點數量 |
| random_seed | number_input | 7 | 控制資料可重現性 |

### 5.5 控制項條件

#### gamma 顯示條件

`gamma` 只在以下 kernel 顯示：

- rbf
- poly
- sigmoid

#### degree 顯示條件

`degree` 只在以下 kernel 顯示：

- poly

### 5.6 Main Sections

#### 5.6.1 Concept Panel

內容：

```text
2D circular data cannot be separated by a straight line.
Kernel methods allow SVM to learn nonlinear decision boundaries.
RBF kernel uses similarity to support vectors to form a flexible boundary.
```

#### 5.6.2 2D Decision Boundary

使用 Plotly Contour + Scatter 顯示：

- 藍色內圈資料點
- 紅色外圈資料點
- `f(x, y) = 0` decision boundary
- `f(x, y) = -1` margin
- `f(x, y) = +1` margin
- support vectors

#### 5.6.3 3D Decision Function Surface

使用 Plotly Surface + Scatter3D 顯示：

- 曲面：

  $$
  z = f(x, y)
  $$

- 訓練資料點放在：

  $$
  z = f(x_i, y_i)
  $$

- support vectors 高亮
- 可以旋轉與縮放
- 可以加入 `z = 0` reference plane

#### 5.6.4 Support Vector Metrics

顯示以下資訊：

- Number of support vectors
- Training accuracy
- Kernel
- C
- Gamma
- Degree（若使用 poly）

#### 5.6.5 Teaching Notes

依照參數動態顯示教學提示：

| 條件 | 顯示訊息 |
|---|---|
| `gamma < 0.2` | Gamma is small: the boundary is smoother and each point has wider influence. |
| `gamma > 3` | Gamma is large: the boundary becomes very flexible and may overfit. |
| `C < 1` | C is small: the model allows more mistakes to keep a wider margin. |
| `C > 20` | C is large: the model tries harder to classify training data correctly. |

### 5.7 Plotly 要求

#### 2D Plot

需要包含：

- Blue inner points
- Red outer points
- Decision boundary contour `f = 0`
- Margin contour `f = -1`
- Margin contour `f = +1`
- Highlighted support vectors

#### 3D Plot

需要包含：

- Decision function surface
- Training points placed at decision score height
- Highlighted support vectors
- `z = 0` reference plane if possible

### 5.8 效能要求

| 項目 | 規格 |
|---|---|
| grid_resolution_default | 80 |
| grid_resolution_max | 150 |
| caching | 使用 `st.cache_data` |
| 避免事項 | 不要使用過高解析度造成 App 卡頓 |

### 5.9 Phase 3 成功標準

- Streamlit App 可用單一指令啟動。
- 使用者可以調整 kernel、C、gamma、degree、noise、number_of_points。
- 2D decision boundary 會正確更新。
- 3D decision function surface 會正確更新。
- Support vectors 顯示正確。
- 介面適合課堂展示。
- 教學文字能幫助初學者理解 SVM 參數。

---

## 6. 共用模組設計

### 6.1 `utils/data_generator.py`

負責資料生成。

必要功能：

1. 產生中心圓形資料點。
2. 產生外圈環狀資料點。
3. 支援 noise。
4. 支援 random seed。
5. 輸出 `X`, `y`。

建議函數：

```python
def generate_ring_dataset(
    n_inner=35,
    n_outer=45,
    inner_radius_range=(0.0, 1.0),
    outer_radius_range=(1.6, 2.5),
    noise=0.08,
    random_seed=7,
):
    pass
```

### 6.2 `utils/svm_utils.py`

負責 SVM 訓練與 decision surface 計算。

必要功能：

1. 訓練 SVC。
2. 產生 decision grid。
3. 計算 decision function。
4. 回傳 support vectors。
5. 支援 kernel 切換。

建議函數：

```python
def train_svm(X, y, kernel="rbf", C=10.0, gamma=1.0, degree=3):
    pass
```

```python
def make_decision_grid(x_range=(-3, 3), y_range=(-3, 3), resolution=100):
    pass
```

```python
def compute_decision_surface(model, grid_points, xx, yy):
    pass
```

---

## 7. 程式碼品質要求

### 7.1 基本要求

- 使用清楚的函數名稱。
- 每個 Phase 可獨立執行。
- 資料生成、模型訓練、視覺化需分離。
- 不要把所有程式碼塞進同一個巨大檔案。
- English code comments。
- 重要教學段落可使用中文 UI 說明。
- 避免過度抽象，保持適合教學與修改。

### 7.2 教學正確性要求

必須避免錯誤說法：

> 不可以說 RBF kernel 就是把資料直接映射到 3D。

正確說法：

> `z = x^2 + y^2` 是教學用的可視化 feature mapping。真正的 RBF kernel 是隱式高維或無限維映射。Phase 2 與 Phase 3 顯示的是 decision function surface，不是完整的 RBF feature space。

### 7.3 視覺品質要求

| 視覺元素 | 要求 |
|---|---|
| Inner class | Blue |
| Outer class | Red |
| Hyperplane / Boundary | Yellow |
| Surface | Translucent |
| Formula | Large and readable |
| Support vectors | Clear ring marker |
| Layout | 適合課堂投影 |

---

## 8. 測試規格

### 8.1 Manual Tests

Agent 實作完成後，需完成以下測試：

1. 執行 Manim 低畫質預覽：

   ```bash
   manim -pql phase1_manim_kernel_trick.py SVMKernelTrick3D
   ```

2. 執行 Manim 高畫質輸出：

   ```bash
   manim -pqh phase1_manim_kernel_trick.py SVMKernelTrick3D
   ```

3. 執行 Phase 2：

   ```bash
   python phase2_rbf_decision_surface.py
   ```

4. 確認 Phase 2 產生：
   - 2D decision boundary
   - 3D decision function surface
   - support vectors

5. 執行 Streamlit App：

   ```bash
   streamlit run phase3_streamlit_app.py
   ```

6. 測試所有 sliders：
   - kernel
   - C
   - gamma
   - degree
   - noise
   - number_of_points
   - random_seed

7. 確認切換 kernel 不會 runtime error。

8. 確認 support vectors 正確顯示。

9. 確認 gamma、C 改變時，decision boundary 會明顯變化。

---

## 9. Agent 實作順序

Antigravity Agent 請依照以下順序實作：

1. 建立 repository structure。
2. 建立 `requirements.txt`。
3. 建立 `utils/__init__.py`。
4. 實作 `utils/data_generator.py`。
5. 實作 `utils/svm_utils.py`。
6. 實作 Phase 1 Manim 動畫：
   - `phase1_manim_kernel_trick.py`
7. 實作 Phase 2 sklearn RBF decision surface：
   - `phase2_rbf_decision_surface.py`
8. 實作 Phase 3 Streamlit interactive app：
   - `phase3_streamlit_app.py`
9. 撰寫 `README.md`。
10. 執行 manual tests。
11. 修正錯誤。
12. 確認最終交付物完整。

---

## 10. Agent 禁止事項

Agent 不可以做以下事情：

1. 不要把所有程式碼合併成單一巨大檔案。
2. 不要錯誤宣稱 RBF kernel 只是 3D mapping。
3. 不要讓 Streamlit App 依賴 Manim 才能執行。
4. 不要使用混亂的隨機顏色。
5. 不要省略 support vectors。
6. 不要省略 README。
7. 不要只做靜態圖，Phase 3 必須可互動。
8. 不要讓圖表缺少標題、公式或教學說明。
9. 不要使用過高 grid resolution 導致效能不佳。
10. 不要忽略 kernel 切換時可能發生的例外狀況。

---

## 11. 最終交付物

完成後，專案應包含：

| 類型 | 檔案 |
|---|---|
| 說明文件 | `README.md` |
| 套件清單 | `requirements.txt` |
| Manim 動畫 | `phase1_manim_kernel_trick.py` |
| RBF SVM 視覺化 | `phase2_rbf_decision_surface.py` |
| Streamlit App | `phase3_streamlit_app.py` |
| 資料工具 | `utils/data_generator.py` |
| SVM 工具 | `utils/svm_utils.py` |
| 輸出資料夾 | `outputs/` |
| 靜態資源資料夾 | `assets/` |

---

## 12. README.md 規格

`README.md` 必須包含以下章節：

1. Project Overview
2. Educational Story
3. Phase 1: Manim Kernel Trick Animation
4. Phase 2: Real RBF SVM Decision Surface
5. Phase 3: Interactive Streamlit Demo
6. Installation
7. Run Commands
8. Important Mathematical Note
9. Teaching Suggestions

### 12.1 README Important Mathematical Note

README 中必須放入以下概念說明：

> The mapping `z = x^2 + y^2` is used as a visual and educational feature mapping to explain why nonlinear data can become linearly separable in a higher-dimensional feature space. A real RBF kernel does not explicitly map data to only 3D; it corresponds to a high-dimensional or infinite-dimensional feature space. Therefore, the RBF decision surface shown in Phase 2 and Phase 3 visualizes the decision function `f(x, y)`, not the full feature space itself.

---

## 13. 驗收標準

### 13.1 功能驗收

| 編號 | 驗收項目 | 是否必須 |
|---|---|---|
| A01 | Manim 動畫可正常渲染 | 必須 |
| A02 | 2D 點雲正確顯示 | 必須 |
| A03 | 3D feature mapping 動畫正確 | 必須 |
| A04 | Hyperplane 正確顯示 | 必須 |
| A05 | 2D decision circle 正確顯示 | 必須 |
| A06 | sklearn RBF SVM 可正常訓練 | 必須 |
| A07 | 2D decision boundary 可視化 | 必須 |
| A08 | 3D decision function surface 可視化 | 必須 |
| A09 | support vectors 被標記 | 必須 |
| A10 | Streamlit App 可正常啟動 | 必須 |
| A11 | sliders 能即時更新圖表 | 必須 |
| A12 | kernel 切換不會錯誤 | 必須 |
| A13 | README 完整 | 必須 |

### 13.2 教學驗收

| 編號 | 驗收項目 | 是否必須 |
|---|---|---|
| T01 | 初學者能理解 2D 非線性不可分 | 必須 |
| T02 | 初學者能理解提升到 3D 後可線性分割 | 必須 |
| T03 | 初學者能理解 hyperplane 與 decision boundary 的關係 | 必須 |
| T04 | 清楚說明 RBF kernel 不是單純 3D mapping | 必須 |
| T05 | 使用者能透過互動操作理解 `C` 與 `gamma` | 必須 |

---

## 14. 建議開發備註

### 14.1 Phase 1 備註

Manim 的 3D 場景要注意：

- `MathTex` 在 3D 場景中可能需要固定在 frame。
- 若公式被 camera 旋轉影響，可使用 fixed in frame mobjects。
- 點的數量不要過多，以免動畫卡頓。
- 曲面透明度不要太高，避免遮住資料點。
- Hyperplane 顏色要明顯，但不要完全遮擋點。

### 14.2 Phase 2 備註

matplotlib 3D 畫圖時要注意：

- Decision surface 的 mesh resolution 不宜過高。
- support vectors 可以用外框高亮。
- 2D 與 3D 圖最好分成兩張圖，避免初學者混淆。
- 圖片標題應清楚標示這是 `decision_function`，不是 RBF feature space。

### 14.3 Phase 3 備註

Streamlit App 要注意：

- 使用 `st.cache_data` 快取資料生成與 grid 計算。
- 每次參數改變時重新 train model。
- 若 kernel 為 linear，gamma 可隱藏。
- 若 kernel 為 poly，degree 需顯示。
- Plotly 3D surface 需控制 resolution，避免瀏覽器卡頓。
- 對學生而言，動態 teaching notes 很重要。

---

## 15. 給 Antigravity Agent 的完整任務說明

請依照本規格書開發一個完整的 SVM Kernel Trick 3D Interactive Demo。

你需要完成三個 Phase：

1. **Phase 1：Manim Concept Animation**
   - 用 Manim 製作 2D 到 3D 的 kernel trick 概念動畫。
   - 使用 `z = x^2 + y^2` 作為教學映射。
   - 顯示 3D hyperplane 與 2D 圓形 decision boundary。

2. **Phase 2：Real RBF SVM Decision Surface**
   - 使用 sklearn `SVC(kernel="rbf")`。
   - 顯示 2D decision boundary、margin、support vectors。
   - 顯示 3D decision function surface。

3. **Phase 3：Interactive Streamlit App**
   - 使用 Streamlit + Plotly。
   - 提供 kernel、C、gamma、degree、noise、number_of_points、random_seed 控制項。
   - 即時更新 2D 與 3D 圖表。
   - 顯示 support vectors 與教學提示。

請務必遵守：

- RBF kernel 不可被錯誤描述為單純 3D mapping。
- 所有檔案需可獨立執行。
- 程式碼需清楚、簡潔、適合教學。
- README 必須完整。
- 最終成果需能用於課堂展示。

---

## 16. 最重要的數學提醒

本專案有兩種視覺化層次：

### 16.1 教學用 3D Mapping

Phase 1 使用：

$$
z = x^2 + y^2
$$

這是為了讓學生直觀看到：

- 中心點比較低
- 外圈點比較高
- 3D 中可以用水平平面切開
- 投影回 2D 是圓形邊界

### 16.2 真正 RBF SVM

Phase 2 與 Phase 3 使用真正的 RBF kernel：

$$
K(x, x') = \exp(-\gamma ||x - x'||^2)
$$

它不是把資料直接丟到簡單 3D 空間，而是使用 kernel trick 隱式計算高維或無限維 feature space 中的內積。

因此，Phase 2 與 Phase 3 的 3D 圖代表：

$$
z = f(x, y)
$$

也就是 SVM 的 decision function surface，而不是完整 feature space。

---

## 17. 完成定義 Definition of Done

本專案完成時，必須符合：

1. 所有指定檔案存在。
2. `pip install -r requirements.txt` 可成功安裝依賴。
3. Manim 預覽指令可成功執行。
4. Phase 2 Python script 可成功產生圖表。
5. Streamlit App 可成功啟動。
6. 所有互動控制項正常運作。
7. Support vectors 正確顯示。
8. README 清楚說明專案目的、安裝、執行、數學注意事項。
9. 程式碼沒有明顯 runtime error。
10. 教學內容對新手友善，且數學敘述正確。

