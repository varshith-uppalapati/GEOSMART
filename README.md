# GEOSMART – Flask Application

A waste-intelligence command centre migrated from Vite/React to **Python Flask**.
The UI, styling, and functionality remain identical to the original React build.

## Project Structure

```
flask_app/
├── app.py                 # Flask backend with all routes & API endpoints
├── requirements.txt       # Python dependencies
├── templates/
│   ├── base.html          # Shared layout (Tailwind CDN, Lucide icons, toast)
│   ├── splash.html        # Splash screen → auto-redirects to login
│   ├── login.html         # Login form
│   ├── signup.html        # Signup form
│   ├── dashboard.html     # Main dashboard with metrics & navigation
│   ├── index_page.html    # Landing / marketing page
│   ├── analyze.html       # Waste stream analyzer with pie chart
│   ├── complaint.html     # Complaint lodge & status board
│   ├── municipal.html     # Municipal officer directory & schedules
│   ├── profile.html       # User profile editor
│   └── 404.html           # Not-found page
└── static/
    ├── css/
    │   └── styles.css     # Design tokens, component styles, animations
    └── js/
        ├── analyze.js     # Client-side waste computation + Chart.js
        ├── complaint.js   # Complaint CRUD via Flask API
        ├── municipal.js   # Officer search / filter
        └── profile.js     # Profile edit & save via Flask API
```

## Prerequisites

- Python 3.9

## Setup & Run

```bash
# 1. Navigate to the project directory
cd flask_app

# 2. Create a virtual environment (recommended)
python -m venv venv

# 3. Activate the virtual environment
#    Windows:
venv\Scripts\activate
#    macOS / Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the application
python app.py
```

The app starts at **http://127.0.0.1:5000**.

## Authentication Modes

The app supports two login paths:

- **User Login:** `/login`
- **Admin Login:** `/admin/login`

Admin credentials are environment-driven (defaults below):

- `ADMIN_EMAIL=admin@geosmart.local`
- `ADMIN_PASSWORD=admin123`

Set custom credentials before running if needed:

```bash
# Windows CMD
set ADMIN_EMAIL=you@domain.com
set ADMIN_PASSWORD=yourStrongPassword
python app.py
```

## Model Training (Admin)

Admins can train the classification model from the dashboard or Analyze page.

1. Login via `/admin/login`
2. Open dashboard training panel
3. Enter dataset YAML path (Ultralytics format)
4. Set epochs and image size
5. Click **Start model training**

On the Analyze page admin panel, you can also:

- Validate a local `data.yaml` path before training
- Upload a dataset `.zip` archive; the app auto-detects `data.yaml` and fills the training path

Additional model APIs:

- `POST /api/model/validate-dataset` (admin)
- `POST /api/model/upload-dataset-zip` (admin)
- `POST /api/analyze/cnn` (authenticated image prediction)

When training finishes, best weights are copied to:

- `models/waste_model.pt`

This model is then used for image verification in complaint uploads.

## Routes

| URL            | Method     | Description                       |
|----------------|------------|-----------------------------------|
| `/`            | GET        | Splash screen (redirects to login)|
| `/login`       | GET / POST | Login page                        |
| `/signup`      | GET / POST | Signup page                       |
| `/dashboard`   | GET        | Main dashboard                    |
| `/index`       | GET        | Landing page                      |
| `/analyze`     | GET        | Waste analysis tool               |
| `/complaint`   | GET        | Complaint desk                    |
| `/municipal`   | GET        | Municipal response hub            |
| `/profile`     | GET        | User profile                      |
| `/logout`      | GET        | Clear session & redirect          |

## API Endpoints

| URL                                  | Method | Description              |
|--------------------------------------|--------|--------------------------|
| `/api/complaints`                    | GET    | List all complaints      |
| `/api/complaints`                    | POST   | Create a complaint       |
| `/api/complaints/<id>/status`        | PATCH  | Update complaint status  |
| `/api/profile`                       | GET    | Get profile data         |
| `/api/profile`                       | PUT    | Update profile data      |

## Technology Stack

- **Backend:** Python Flask
- **Styling:** Tailwind CSS (CDN) with custom HSL design tokens
- **Icons:** Lucide (CDN)
- **Charts:** Chart.js (CDN)
- **Templating:** Jinja2

## Paper Alignment (GeoSmart)

This codebase implements the core workflow described in your paper:

- **Citizen reporting with image + GPS:** Complaint form supports evidence upload and location capture.
- **AI-based verification:** `POST /api/analyze/cnn` verifies uploaded images and returns predicted waste type + confidence.
- **Hotspot detection (GIS-aware):** Hotspots are generated from complaint geolocation with severity scoring and queue ranking.
- **Route optimization (A*):** Municipal hub calls `GET /api/routes/optimize` to generate shortest dispatch plans from nearest depot.
- **Live dashboards:** Dashboard, complaint desk, and municipal hub auto-refresh metrics/hotspots periodically.
- **Lifecycle tracking:** Complaint status flow is aligned to: `Pending -> Assigned -> In Progress -> Resolved`.
- **Admin model training:** Dataset validation, zip upload, and model retraining endpoints are available from dashboard/analyze pages.

If you want to match the paper further, the next optional additions are:

- **IoT bin ingestion API** for sensor streams.
- **True multi-stop VRP optimizer** beyond single-hotspot route selection.
- **Traffic-aware ETA integration** using an external roads/traffic API.
