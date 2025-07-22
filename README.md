```
nb_classifier_project/
│
├── 📁 service_data_loader/
│   ├── 📁 app/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 main_api.py           # ה-API של טעינת הנתונים (POST /load)
│   │   └── 📄 data_loader.py        # לוגיקת הטעינה (הקוד הקיים שלך)
│   ├── 📄 Dockerfile
│   └── 📄 requirements.txt          # (fastapi, uvicorn, pydantic, pandas, openpyxl)
│
├── 📁 service_preprocessor/
│   ├── 📁 app/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 main_api.py           # ה-API של העיבוד (POST /clean, POST /split)
│   │   ├── 📄 cleaner.py            # לוגיקת הניקוי (הקוד הקיים)
│   │   └── 📄 splitter.py           # לוגיקת הפיצול (הקוד הקיים)
│   ├── 📄 Dockerfile
│   └── 📄 requirements.txt          # (fastapi, uvicorn, pydantic, pandas, scikit-learn)
│
├── 📁 service_model_builder/
│   ├── 📁 app/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 main_api.py           # ה-API של בניית המודל (POST /build)
│   │   ├── 📄 builder.py            # לוגיקת בניית המודל (הקוד שעבדנו עליו)
│   │   └── 📄 typing_defs.py        # הגדרות הטיפוסים הספציפיות למודל
│   ├── 📄 Dockerfile
│   └── 📄 requirements.txt          # (fastapi, uvicorn, pydantic, pandas)
|
├── 📁 service_evaluator/
│   ├── 📁 app/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 main_api.py       # ה-API של הערכת המודל (POST /evaluate)
│   │   ├── 📄 evaluator.py      # לוגיקת חישוב המדדים (accuracy, precision, etc.)
│   │   └── 📄 typing_defs.py    # צריך להכיר את מבנה המודל
│   ├── 📄 Dockerfile
│   └── 📄 requirements.txt      # (fastapi, uvicorn, pydantic, pandas, scikit-learn)
|
├── 📁 service_inference/
│   ├── 📁 app/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 main_api.py           # ה-API של ההיסק (POST /predict, GET /schema)
│   │   ├── 📄 predictor.py          # לוגיקת החיזוי (דורש refactoring ל-OOP)
│   │   └── 📄 typing_defs.py        # עותק של הגדרות הטיפוסים (כדי להבין את מבנה המודל)
│   ├── 📄 Dockerfile
│   └── 📄 requirements.txt          # (fastapi, uvicorn, pydantic, pandas)
│
├── 📁 shared_artifacts/
│   └── 📄 model.json                # (נוצר על ידי המנצח, נקרא על ידי service_inference)
│
├── 📁 data/
│   └── 📄 dataset.csv               # קובץ הנתונים הגולמי
│
├── 📄 docker-compose.yml            # קובץ התזמור שמריץ את כל השירותים
├── 📄 orchestrator.py               # סקריפט שמפעיל את פייפליין האימון
├── 📄 .gitignore
└── 📄 README.md
```
