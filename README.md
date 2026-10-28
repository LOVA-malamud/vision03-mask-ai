# VISION · Face Mask CNN

An educational image-classification application connecting a TensorFlow/Keras CNN to a FastAPI backend and a React/TypeScript interface. The UI and detailed learning guide are in Hebrew.

**Status:** model and application evidence are recorded; physical-phone camera/gallery/save acceptance remains pending. No public deployment is claimed.

[Hebrew guide](#hebrew-guide) · [Training notebook](notebooks/Project03_VISION_HE.ipynb) · [Verification](docs/VERIFICATION.md) · [Final review](docs/FINAL_REVIEW_HE.md)

## What to explore

- A binary classifier for whole input images: with mask / without mask.
- Shared preprocessing between training and inference, and validation of the model/metadata pair.
- Upload validation, a Hebrew RTL interface, and backend tests.
- Reproducible training notes and recorded evaluation artifacts under [reports](reports/).

The classifier does not detect faces or assess correct mask placement. Recorded dataset metrics are not evidence of real-world deployment performance.

## Local setup

The existing guide records Python 3.12 on Intel macOS; use Python 3.10–3.12 and Node.js 22+. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
npm --prefix frontend ci
npm --prefix frontend run build
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8001
```

Open http://127.0.0.1:8001 after application startup completes. The bundled model is used without retraining. For checks, run `python -m pytest -q` in that environment. These commands were checked against repository paths and manifests; dependencies, TensorFlow inference and physical-device behavior were not rerun for this documentation change. [Verification](docs/VERIFICATION.md) and [final model validation](reports/final_model_validation.json) retain the recorded results and limits.

<a id="hebrew-guide"></a>

<details>
<summary>Full Hebrew guide · מדריך לימודי מלא</summary>

# Project 03 · VISION — Face Mask CNN

יישום לימודי שמסווג **תמונת פנים שלמה** ל־**עם מסכה** או **ללא מסכה**. React + TypeScript (ממשק עברי RTL), FastAPI, ו־CNN קטן שנבנה ואומן מאפס ב־TensorFlow/Keras. התחזיות אמיתיות; אין שירות AI חיצוני או מפתח API.

[מאגר GitHub](https://github.com/LOVA-malamud/vision03-mask-ai) · [פתחו את המחברת ב־Colab](https://colab.research.google.com/github/LOVA-malamud/vision03-mask-ai/blob/main/notebooks/Project03_VISION_HE.ipynb) · [דוח ביקורת ודרישות](docs/FINAL_REVIEW_HE.md)

**מצב ההגשה:** תוצרי הקוד והמודל נבדקו. אישור סופי עדיין ממתין לבדיקת מצלמה/גלריה/שמירה בטלפון פיזי ולבדיקת הורדה חוזרת. המייל לא נשלח.

## הפעלה מקומית

נבדק ב־Python 3.12 על Intel macOS. נתמכות גרסאות Python 3.10–3.12; מומלץ 3.12. נדרש Node.js 22+ לבניית הממשק. אין צורך לאמן כדי להפעיל את המודל המצורף.

```bash
git clone https://github.com/LOVA-malamud/vision03-mask-ai.git
cd vision03-mask-ai
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
cd frontend
npm ci
npm run build
cd ..
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8001
```

פתחו http://127.0.0.1:8001. תיעוד ה־API: http://127.0.0.1:8001/docs.
ב־Windows: `py -3.12 -m venv .venv`, ואז `.venv\Scripts\activate` במקום `source`.
`backend/requirements.lock.txt` שומר את גרסאות סביבת ההסקה המקומית שנבדקה; סביבת האימון ב־Colab מתועדת בנפרד ב־reports/training_environment.json.

בהפעלה הראשונה אחרי התקנה, טעינת TensorFlow עשויה להימשך כמה דקות. המתינו לשורה `Application startup complete` לפני פתיחת האתר.

ב־macOS אפשר גם לפתוח את `Start.command` לאחר התקנת התלויות. לעצירה: Ctrl+C בטרמינל.
לעריכת הממשק: הפעילו את השרת ובטרמינל נוסף `cd frontend && npm run dev` (פורט 5174).

## שימוש

1. בחרו **גלריה / קובץ**, גררו תמונה, או בחרו **צילום תמונה** בטלפון.
2. בדקו את התצוגה המקדימה. JPEG/PNG/WebP (כולל התמונה הראשית ב־MPO JPEG), עד 10 MB ועד 20 מגה־פיקסל. HEIC יש להמיר ל־JPEG.
3. **שמור תמונה** מוריד את הקובץ המקורי למכשיר בדפדפן תומך הורדות (מומלץ Chrome); הפעולה אינה מסווגת ואינה מוסיפה תמונה למאגר אימון.
4. **בצע סיווג** שולח את התמונה ל־FastAPI ומציג תווית ושתי הסתברויות. השרת אינו שומר את ההעלאה.

צילום משתמש ב־`input type=file accept=image/* capture=user`. במחשב נפתח בדרך כלל בוחר קבצים. התמיכה המדויקת במצלמה תלויה בדפדפן ובטלפון. לבדיקת טלפון באותה רשת, הפעילו עם `--host 0.0.0.0` וגשו ל־`http://<כתובת-המחשב>:8001`; `localhost` בטלפון אינו המחשב. לא הוגדרו ngrok או אתר ציבורי.

## נתונים, אימון ותוצאות

מקור: [Face Mask Dataset — Shiekh Burhan, Kaggle, version 1](https://www.kaggle.com/datasets/shiekhburhan/face-mask-dataset). לפי מטא־הנתונים ב־Kaggle הרישיון CC0. המאגר עצמו אינו מצורף להגשה; שתי דוגמאות מבחן עם מקורן מצורפות לבדיקות.

מבנה שנבדק בפועל:

| תיקייה | simple | complex | סך הכול |
|---|---:|---:|---:|
| with_mask | 4,000 | 789 | 4,789 |
| without_mask | 4,000 | 747 | 4,747 |
| incorrect_mask | mc: 2,500 | mmc: 2,500 | 5,000 — לא בשימוש |

אין במקור חלוקה מובנית לאימון/מבחן. `simple`/`complex` אינם תוויות. קראנו 9,536 קבצים משתי המחלקות, הסרנו 741 כפילויות לאחר עיבוד, ודילגנו על תמונה אחת מעל 20 מגה־פיקסל. נשארו 8,794 תמונות. קבוצות עם dHash זהה אינן מתפצלות בין קבוצות הנתונים; אין הבטחת הפרדה לפי זהות אדם.

| חלוקה | עם מסכה | ללא מסכה | סך הכול |
|---|---:|---:|---:|
| אימון | 3,302 | 2,853 | 6,155 |
| ולידציה | 724 | 594 | 1,318 |
| מבחן | 713 | 608 | 1,321 |

<!-- ACTUAL_RESULTS -->
תוצאות אמיתיות של האימון שהושלם בחשבון Colab ב־T4 GPU, גרסה `20261017T175219Z`:

- **דיוק מבחן: 94.17%** על 1,321 תמונות שהופרדו מראש; loss: 0.1541.
- דיוק ולידציה של המודל שנבחר: 95.60%; val_loss: 0.1240.
- 20 אפוקים בפועל; checkpoint שנבחר: אפוק 16.
- מטריצת בלבול, שורות=אמת ועמודות=תחזית, בסדר `[with_mask, without_mask]`: `[[670, 43], [34, 574]]`.
- הפרש מרבי בין תחזיות לפני ואחרי טעינה מחדש: `0.0` (8 תמונות ולידציה).
- קבצים מלאים: `reports/evaluation.json`, `history.csv`, `training.png`, `confusion_matrix.png`, ו־`dataset_manifest.json`.
<!-- END_RESULTS -->

המבנה לפי מחברת הכיתה: Conv2D(32,3) → MaxPool → Conv2D(32,3) → MaxPool → Flatten → Dense(128,ReLU) → Dense(1,Sigmoid), ובסך הכול 813,217 פרמטרים. נוספו Flip/Zoom ו־Dropout באימון בלבד. Adam, binary crossentropy, batch 32, עד 20 אפוקים. בחירת checkpoint לפי **val_loss**, עם EarlyStopping והקטנת learning rate; המבחן משמש להערכה רק לאחר בחירת המודל וטעינתו מחדש.

המחברת `notebooks/Project03_VISION_HE.ipynb` עצמאית, כוללת הסברים בעברית ופלטי האימון האמיתי ב־Colab. לאחר ניתוק הסביבה שוחזר המודל מ־Drive ללא אימון נוסף. ביטול בחירת תמונה מציג הנחיה להריץ שוב רק את תא התחזית. [הוראות Colab המדויקות](docs/COLAB_HE.md).

## חוזה אימון ↔ אתר

`backend/app/preprocessing.py` משותף לאימון ולשרת: EXIF → RGB → Pillow bilinear 64×64 → float32 / 255 → batch. אין נרמול כפול. `with_mask=0`, `without_mask=1`; Sigmoid הוא `P(without_mask)`, וסף ההחלטה 0.5.

`models/mask_cnn.keras` ו־`models/metadata.json` הם זוג מחייב. FastAPI טוען אותם פעם אחת ב־lifespan, בודק SHA-256, מיפוי, עיבוד וצורת קלט/פלט, ומשתמש באותו מודל לכל בקשה. החלפתם מחייבת הפעלה מחדש. אין fallback לתחזית מדומה. `/api/health` מדווח אם המודל נטען.

## שחזור ובדיקות

```bash
# אחרי הורדת ZIP של גרסה 1 מ־Kaggle:
python -m ml.prepare /path/to/face-mask-dataset.zip
# תיקיית פלט חדשה — לא דורסים מודל קיים:
python -m ml.train --prepared data/prepared --output runs/new/models --reports runs/new/reports
python -m pytest -q
```

אפשר לבצע כל תא במחברת מקומית ולשמור פלטים אמיתיים באמצעות `python -m pip install ipython`, ואז `VISION_DATASET_ZIP=/path/to/dataset.zip python scripts/execute_notebook.py`. המחברת משתמשת בנתונים מוכנים רק אם checksum הארכיון זהה, ויוצרת תיקיית ריצה חדשה אם כבר קיים מודל.

כל 16 הבדיקות עברו, כולל MPO מתמונת טלפון ודחיית תמונות מונפשות. הבדיקות משוות העלאת HTTP לתחזית ישירה של המודל שנטען מהדיסק, בשתי מחלקות, בודקות טעינה יחידה, קלטים פגומים, מגבלת גודל ומיפוי שגוי. ראו [דוח אימות](docs/VERIFICATION.md).

## הגשה ומגבלות

יש להעלות ל־GitHub את `backend`, `frontend` (קוד ומנעול תלויות), `ml`, `models` **כולל `.keras`**, `notebooks`, `reports`, `tests`, `docs` וה־README. `.gitignore` אינו מסנן את המודל. אין להעלות `.venv`, `node_modules`, תמונות המאגר או סודות. כתובת המאגר מופיעה בראש הדף. המייל למרצה לא נשלח; סטטוס האימות מופיע בדוח הביקורת.

המודל תמיד בוחר אחת משתי מחלקות. הוא אינו מזהה אם יש פנים, כמה אנשים בתמונה או אם המסכה נלבשת כראוי. חלק מדוגמאות המקור עשויות להיות רועשות/עמומות, ואין מזהי אנשים למניעת כל דמיון בין פיצולים. המדדים תקפים לפיצול זה של המאגר ואינם מחקר שטח או כיול הסתברויות. שינוי תאורה, מצלמה, רקע, כיוון פנים או קבוצת אוכלוסייה עלול לפגוע בדיוק. Unknown מתחת ל־60%, ngrok ופריסה ציבורית לא יושמו.

הפרויקט המקורי נשמר בנפרד ללא שינוי. [בדיקת המקור והשימוש החוזר](docs/ORIGINAL_AUDIT.md) · [הסבר קצר להצגה](docs/PRESENTATION_HE.md).

מקורות הקורס: [מטלה מלאה](https://pythonai211225-rgb.github.io/advanced_ai/project03_cnn.pdf), [הסבר CNN](https://pythonai211225-rgb.github.io/advanced_ai/02_cnn/index.html), [מחברת הכיתה](https://github.com/pythonai211225-rgb/11.10.2026/blob/main/cnn.ipynb).

תיקון תאימות לאחר האימון: נוספה תמיכה ב־MPO JPEG בשרת ובמחברת, בלי לשנות משקלים או מדדים. פלטי המחברת נשמרו מהאימון המקורי ולא נוצרו מחדש. הסיווג באתר נבדק עם תמונת המשתמש; הורדת אותה תמונה מהדפדפן המשובץ ב־Codex לא אושרה, ולכן בדיקת שמירה חוזרת ב־Chrome נותרה פעולה ידנית.

## אימות ואבטחת תלויות — 21.10.2026

המודל אומן ב־Keras 3.12.0, ונבדק מחדש ב־Keras 3.15.1 ו־Pillow 13.0.0 לאחר עדכוני אבטחה. המשקלים המקוריים לא השתנו. `reports/final_model_validation.json` מתעד הערכה חוזרת על כל 1,321 תמונות המבחן. `reports/python_audit.json` ו־`reports/npm_audit.json` מכילים את הסריקות; נותרה התראת Protobuf אחת שאינה נמצאת בנתיב הקלט של היישום, כמפורט בדוח הביקורת. אין טענה למוכנות לשירות ציבורי קבוע.

גודל המודל כ־9.8 MB, ולכן הוא כלול ישירות ב־Git ללא צורך ב־Git LFS. הנתונים המלאים אינם כלולים; שתי תמונות בדיקה ציבוריות בלבד מצורפות עם נתיבי המקור שלהן.

</details>
