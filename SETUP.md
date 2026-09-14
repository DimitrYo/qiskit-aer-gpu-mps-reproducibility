# Середовище виконання

Для перевірки записаних даних та графіків достатньо Python і `requirements.txt`;
короткі команди наведено в [README.md](README.md). Нижче — середовище для нових
числових експериментів на Linux або Ubuntu у WSL2.

## 1. Системні програми

Якщо у Windows ще немає WSL2, відкрийте PowerShell від адміністратора:

```powershell
wsl --install -d Ubuntu-24.04
```

Перезавантажте Windows, якщо це запропонує інсталятор, і створіть користувача
в Ubuntu. Перевірте версію WSL і відкрийте Linux-термінал:

```powershell
wsl --list --verbose
wsl -d Ubuntu-24.04
```

У списку для Ubuntu має бути `VERSION 2`; для наявної інсталяції WSL1
виконайте в PowerShell `wsl --set-version Ubuntu-24.04 2`
([офіційна інструкція Microsoft](https://learn.microsoft.com/en-us/windows/wsl/install)).
Подальші команди виконуйте в Ubuntu bash. Клонуйте репозиторій у домашній
каталог Linux і перейдіть до нього перед створенням Python-середовища.

Рецепт розрахований на Ubuntu 24.04 x86-64. У записаній збірці використано
CPython **3.11.15**, GCC **13.3.0**, CMake **3.28.3**, CUDA Toolkit **12.9**
(`nvcc 12.9.86`), OpenBLAS, nlohmann-json **3.11.3** і spdlog **1.12.0**.
Первинні журнали та параметри CMake збережено в `backend/recorded-builds.zip`.

```bash
sudo apt update
sudo apt install -y git build-essential gcc-13 g++-13 cmake ninja-build \
  libopenblas-dev liblapack-dev nlohmann-json3-dev libspdlog-dev python3-pip python3-venv
export CC=gcc-13
export CXX=g++-13
```

Для GPU встановіть драйвер NVIDIA та **CUDA Toolkit 12.9** за
[інструкцією NVIDIA для Linux](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-installation-guide-linux/index.html).
Після підключення відповідного репозиторію NVIDIA пакет встановлюється командою
`sudo apt install cuda-toolkit-12-9`.
Для Windows використовуйте WSL2: драйвер встановлюється у Windows,
а Toolkit — усередині WSL; Linux-драйвер у WSL встановлювати не слід
([інструкція NVIDIA для WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)).

```bash
nvidia-smi
/usr/local/cuda/bin/nvcc --version
gcc-13 --version
cmake --version
```

Якщо Toolkit розташований у `/usr/local/cuda-12.9`, додайте до команд збірки
`--cuda-root /usr/local/cuda-12.9`. Архітектура записаної збірки — **89**
(compute capability 8.9); для іншого GPU вкажіть його значення через `--cuda-arch`.

## 2. Python і бібліотеки

Нижче Python 3.11.15 встановлюється через
[uv](https://docs.astral.sh/uv/guides/install-python/). Виконуйте команди
в корені клонованого репозиторію. Середовище `.venv-gpu` має бути новим.

```bash
python3 -m venv .bootstrap
.bootstrap/bin/python -m pip install uv
.bootstrap/bin/uv python install 3.11.15
.bootstrap/bin/uv venv --python 3.11.15 --seed .venv-gpu
source .venv-gpu/bin/activate
python -m pip install -r requirements-gpu.txt
python -m pip check
```

Основні версії: Qiskit 2.5.0, NumPy 2.4.6, threadpoolctl 3.6.0,
cuQuantum 24.8.0, cuStateVec 1.6.0.post1, cuTensorNet 2.5.0, cuTENSOR 2.7.0.
`requirements-gpu.txt` поєднує вимоги протоколу і версії, зафіксовані на локальному
комп'ютері 13 вересня 2026 року. Це рецепт відновлення середовища, а не повний
історичний lock усіх системних і транзитивних залежностей.

Модифікація GPU MPS входить до включених джерел Aer. Пакет
`pip install qiskit-aer-gpu` не є заміною цієї збірки.

## 3. Збірка Aer

```bash
python build_backend.py --backend custom --check-only
python build_backend.py --backend custom
python build_backend.py --backend upstream
```

Скрипт бере джерела з `backend/aer-history.bundle`, компілює їх CMake/Ninja
і перевіряє завантаження Python-модуля. Фіксовані ревізії:

| Варіант | Git commit | Результат |
| --- | --- | --- |
| Custom CPU/GPU MPS | `a1242579272784330b217085fe11b7e1481922ad` | `.build/custom/backend.json` |
| Upstream CPU MPS | `51c679814c3a292d0d7c59bb39976bd6ff91f60e` | `.build/upstream/backend.json` |
| Historical CPU/GPU MPS (необов'язкова) | `75137ba7dee76ec864cb723a6c3aa9650fbf4f43` | `.build/historical/backend.json` |

Обидві збірки використовують одне Python-середовище; їхні модулі завантажуються
в окремих процесах із відповідного каталогу джерел. Збірка за замовчуванням
використовує два паралельні процеси (`--jobs 2`). Логи компіляції й помилок
зберігаються поруч із `backend.json`. Для повторної збірки оберіть новий каталог
через `--output`, наприклад `--output .build/custom-second`.

`--check-only` перевіряє наявність інструментів і заголовків без компіляції.
Успішна перевірка не означає, що GPU-обчислення вже перевірені.
Для окремих історичних порівнянь виконайте також
`python build_backend.py --backend historical`; основні експерименти
використовують `custom` і `upstream`.

## 4. Запуск

Єдиний seed нових запусків — `20260912` (константа `MASTER_SEED` у
`run_experiment.py`). Підготовка зберігає політику seed, нові хеші схем і
адаптації робочих копій скриптів. Старі архіви не змінюються.

Поверніться до команд у [README.md](README.md). Вимірювання потребують шести
доступних CPU та вільного GPU. Запускайте етапи послідовно, уникаючи інших
обчислень на цих самих ресурсах. Архіви `data/` містять записані параметри,
схеми, seeds і початкові результати; нові виміри пишуться до `runs/`.

Native Windows підтримує роботу зі збереженими даними та графіками. Сценарії
нового виконання використовують Linux CPU affinity і `.so`-модулі, тому для
них потрібен Linux/WSL2.
