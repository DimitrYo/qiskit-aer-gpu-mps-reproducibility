# Відтворення результатів GPU MPS у Qiskit Aer

Тут зібрано код запуску експериментів, середовище виконання, початкові логи
статті та скрипт побудови графіків. Для перегляду результатів NVIDIA GPU не потрібен.
Для повторного вимірювання потрібні Linux/WSL2, NVIDIA GPU та зібрана з включених
джерел модифікація Aer.

## Перевірити дані та побудувати графіки

Встановіть Git і Python 3.11 або новіший. У Linux/macOS:

```bash
git clone https://github.com/DimitrYo/qiskit-aer-gpu-mps-reproducibility.git
cd qiskit-aer-gpu-mps-reproducibility
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verify_results.py
python plot_results.py
```

У Windows використайте `py -3.11 -m venv .venv` і запускайте команди через
`.venv\Scripts\python.exe` замість `python`. Графіки з'являться в `output/figures/`.
Ці команди перевіряють та обробляють збережені дані; вони не збирають нові виміри.
Скрипт створює два рисунки поточної редакції статті у PNG, SVG і PDF,
таблицю 1 з умовами та `results.csv` із числовими результатами. Відповідність
методики, висновків і графіків конкретним файлам наведено в [ARTICLE.md](ARTICLE.md).

Для шести рисунків старої редакції: `python plot_results.py --edition 020 --output-dir output/figures-020`.

## Повторити експерименти

Усі нові запуски використовують **seed `20260912`**: генерація схем,
симулятор, транспілятор, порядок запусків і bootstrap. Незалежні входи отримують
окремі потоки `PCG64(20260912).jumped(stream_id)`; їхні номери записуються разом
із кутами схем. Це зберігає різні входи та відокремлює пілот від підтвердження.
Використайте новий `--run-root`: каталоги зі старою політикою seed відхиляються.

Зміна seed створює нові входи й контрольні суми. Архівні логи та перевірка
результатів статті зберігають фактичні початкові seed; `verify_results.py` і
`plot_results.py` відтворюють саме ці записані результати.

Спочатку виконайте інструкції [SETUP.md](SETUP.md): вони містять установку Python,
компілятора, CUDA, бібліотек і точні команди збірки обох варіантів Aer.
Після збірки запустіть з активованого середовища:

```bash
python run_experiment.py prepare --custom-build .build/custom/backend.json --upstream-build .build/upstream/backend.json --affinity 0,2,4,6,8,10 --run-root runs/reproduction
python run_experiment.py smoke --run-root runs/reproduction
python run_experiment.py cpu-selection --run-root runs/reproduction
python run_experiment.py pilot --run-root runs/reproduction
python run_experiment.py confirmation --run-root runs/reproduction
```

Етап `confirmation` збирає першу сесію і завершується станом `PAUSED`.
Після її завершення залиште той самий комп'ютер без експериментів щонайменше
на 30 хвилин, потім виконайте другу сесію й аналіз:

```bash
python run_experiment.py resume --run-root runs/reproduction
python run_experiment.py validate --run-root runs/reproduction --study confirmation
```

Окремий експеримент із 26 фіксованими входами запускається після `prepare`:

```bash
python run_experiment.py fixed-prepare --run-root runs/reproduction
python run_experiment.py fixed-probe --run-root runs/reproduction
python run_experiment.py fixed-input --run-root runs/reproduction
python run_experiment.py validate --run-root runs/reproduction --study fixed-input
```

`fixed-probe` перевіряє ресурсну доступність і коректність перед повним запуском.
Якщо виконання перервано, використайте `fixed-resume` з тим самим `--run-root`.

Для додаткового історичного порівняння сімейств схем (рисунок 6 редакції 020) використовується окрема
зафіксована версія Aer і початкові скрипти з `backend/historical-runners.zip`:

```bash
python build_backend.py --backend historical
python run_experiment.py historical-workloads --run-root runs/reproduction --historical-build .build/historical/backend.json
```

Це повний історичний набір із 32 входів. Для історичного масштабування використайте
ту саму команду з `--historical-scope scaling16`, `scaling18` або `scaling20`.
Нові журнали пишуться окремо від основних експериментів у каталозі запуску.

Список `--affinity` замініть на шість доступних процесорів свого комп'ютера;
перевірте їх через `lscpu -e=CPU,CORE,ONLINE`. Команди виконуються послідовно:
вибір CPU-конфігурації визначає умови подальших порівнянь.
Нові логи зберігаються в `runs/reproduction/`. Докладні параметри кожної команди:
`python run_experiment.py <команда> --help`.

Час виконання залежить від GPU, CPU, частот і фонових процесів. Повторний запуск
має відтворити протокол і перевірки коректності; ідентичні секунди не гарантуються.

## Що міститься в репозиторії

| Файл або папка | Призначення |
| --- | --- |
| `run_experiment.py` | Підготовка та повторний запуск вимірювань |
| `build_backend.py` | Збірка зафіксованих поточної, upstream і історичної ревізій Aer |
| `verify_results.py`, `plot_results.py` | Перевірка записаних даних і графіки |
| `SETUP.md`, `requirements*.txt` | Програми, версії та команди встановлення |
| `ARTICLE.md` | Зв’язок методики й результатів поточної статті з кодом та логами |
| `data/` | Архіви початкових логів, параметрів і точних скриптів запуску |
| `backend/` | Джерела Aer, ліцензія та записані журнали збірки |
| `PROVENANCE.json`, `data/index.json` | Походження та перелік включених даних |

Архіви зберігають початкові логи без редагування. У них можуть бути старі локальні
шляхи та службові назви запусків: вони потрібні для перевірки походження,
а нові команди самостійно підставляють шляхи вашого комп'ютера.
