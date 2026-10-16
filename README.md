# FitAI - AI-Powered Fitness App

A full-stack fitness application with AI workout planning, diet recommendations, real-time pose monitoring, food scanning, an AI chatbot, and gamification.

## Tech Stack
- **Backend**: Flask (Python)
- **Database**: SQLite
- **AI**: Google Gemini API
- **Pose Detection**: MediaPipe (browser-based)
- **Frontend**: HTML/CSS/JavaScript

## Features
1. **AI Workout Planner** - Personalized weekly workout plans via Gemini API
2. **AI Diet Planner** - Custom meal plans with macro breakdown
3. **Workout Monitor** - Real-time rep counting + form feedback via MediaPipe Pose
4. **Food Scanner** - AI-powered food image analysis with nutrition info
5. **AI Chatbot** - Conversational fitness coach via Gemini API
6. **Gamification** - XP, badges, streaks, leaderboard

## Setup

### 1. Install Dependencies
```bash
pip install flask google-generativeai
```

### 2. Set Gemini API Key
Get your free API key from https://aistudio.google.com/

Option A - Environment variable (recommended):
```bash
export GEMINI_API_KEY=your_api_key_here
python app.py
```

Option B - Edit app.py line 12:
```python
GEMINI_API_KEY = 'your_api_key_here'
```

### 3. Run the App
```bash
cd fitness_app
python app.py
```

Open browser: http://localhost:5000

## Project Structure
```
fitness_app/
├── app.py              # Flask backend (all API routes)
├── requirements.txt
├── fitness.db          # SQLite DB (auto-created)
├── uploads/            # Uploaded images
├── static/
│   ├── css/style.css   # All styles
│   └── js/app.js       # Global JS
└── templates/
    ├── base.html       # Sidebar + navbar layout
    ├── index.html      # Landing page
    ├── dashboard.html  # Stats overview
    ├── workout.html    # AI workout planner
    ├── diet.html       # AI diet planner
    ├── monitor.html    # Webcam pose monitor
    ├── nutrition.html  # Food scanner
    ├── chatbot.html    # AI chatbot
    └── gamification.html # XP, badges, leaderboard
```

## API Routes
| Method | Route | Description |
|--------|-------|-------------|
| POST | /api/user/setup | Create/update user profile |
| GET | /api/user/:id | Get user data |
| POST | /api/workout/recommend | Generate AI workout plan |
| POST | /api/diet/recommend | Generate AI diet plan |
| POST | /api/nutrition/analyze | Analyze food image |
| POST | /api/chat | Chatbot message |
| GET | /api/gamification/:id | Get XP/badges/leaderboard |
| POST | /api/gamification/log_workout | Log workout + earn XP |

## Notes
- The workout monitor uses MediaPipe from CDN (requires internet)
- SQLite DB is created automatically on first run
- Food scanner uses Gemini's vision capabilities
- All AI features require a valid Gemini API key
