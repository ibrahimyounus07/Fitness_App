from flask import Flask, render_template, request, jsonify, session
from groq import Groq
import sqlite3
import os
import json
import base64
from datetime import datetime, date
import uuid
import io
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

# ─── AUTO-LOAD .env FILE ───────────────────────────────────────────────────
def _load_env_file():
    for filename in ('.env', '.env.example'):
        env_file = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(env_file):
            try:
                with open(env_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            k, v = line.split('=', 1)
                            k, v = k.strip(), v.strip().strip('"').strip("'")
                            if k and (k not in os.environ or not os.environ[k]):
                                os.environ[k] = v
            except Exception:
                pass

_load_env_file()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'fitness-app-secret-key-2024')

# Configure Groq API (free tier: ~14,400 requests/day)
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
GROQ_TEXT_MODEL = os.environ.get('GROQ_TEXT_MODEL', 'openai/gpt-oss-20b')
GROQ_VISION_MODEL = os.environ.get('GROQ_VISION_MODEL', 'openai/gpt-oss-20b')

def get_groq_client():
    global GROQ_API_KEY
    if not GROQ_API_KEY:
        _load_env_file()
        GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
    if GROQ_API_KEY:
        return Groq(api_key=GROQ_API_KEY)
    return None

class _GroqClientProxy:
    """Safely proxies calls to Groq client or raises an informative error if key is missing."""
    def __getattr__(self, item):
        client = get_groq_client()
        if client is None:
            raise RuntimeError("Groq API key is missing. Please add GROQ_API_KEY to your .env file.")
        return getattr(client, item)

groq_client = _GroqClientProxy()

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ─── LOAD TRAINED FOOD MODEL ─────────────────────────────────────────────────

_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
_FOOD_MODEL_PATH = os.path.join(_MODEL_DIR, 'food_classifier.keras')
_CLASS_NAMES_PATH = os.path.join(_MODEL_DIR, 'class_names.json')

food_classifier = tf.keras.models.load_model(_FOOD_MODEL_PATH)
with open(_CLASS_NAMES_PATH, 'r') as _f:
    FOOD_CLASS_NAMES = json.load(_f)

IMG_SIZE = (224, 224)  # MobileNetV2 input size


def predict_food(img_bytes: bytes):
    """Fallback: run the trained MobileNetV2 classifier on raw image bytes.
    The primary detection is now handled by the vision API; this helper is only used when
    the external API fails or returns an unexpected result.
    Returns (food_name: str, confidence: float, all_top5: list)."""
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = mobilenet_preprocess(arr)          # scale to [-1, 1]
    arr = np.expand_dims(arr, axis=0)        # (1, 224, 224, 3)
    preds = food_classifier.predict(arr, verbose=0)[0]  # (num_classes,)
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx])
    food_name = FOOD_CLASS_NAMES[top_idx]
    # Top-5 for debugging / logging
    top5_idx = np.argsort(preds)[::-1][:5]
    top5 = [{'name': FOOD_CLASS_NAMES[i], 'prob': float(preds[i])} for i in top5_idx]
    return food_name, confidence, top5

# ─── DATABASE SETUP ──────────────────────────────────────────────────────────

def get_db():
    db = sqlite3.connect('fitness.db')
    db.row_factory = sqlite3.Row
    return db

def init_db():
    db = get_db()
    db.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            role TEXT DEFAULT 'user',            password_hash TEXT,            email TEXT,
            age INTEGER,
            gender TEXT,
            weight REAL,
            height REAL,
            bmi REAL,
            fitness_goal TEXT,
            activity_level TEXT,
            food_preference TEXT,
            target_weight REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS workouts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            workout_type TEXT,
            duration INTEGER,
            calories_burned INTEGER,
            completed_at TEXT DEFAULT CURRENT_TIMESTAMP,
            xp_earned INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS gamification (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT UNIQUE,
            total_xp INTEGER DEFAULT 0,
            level INTEGER DEFAULT 1,
            streak_days INTEGER DEFAULT 0,
            last_login TEXT,
            total_workouts INTEGER DEFAULT 0,
            badges TEXT DEFAULT '[]'
        );

        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            role TEXT,
            message TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS weight_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            weight REAL,
            log_date TEXT,
            notes TEXT,
            logged_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS workout_completions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            completion_date TEXT,
            completed INTEGER DEFAULT 0,
            workout_focus TEXT,
            duration INTEGER DEFAULT 0,
            notes TEXT,
            logged_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS diet_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            plan TEXT,
            plan_start_date TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS meal_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            log_date TEXT,
            meal_index INTEGER,
            meal_name TEXT,
            completed INTEGER DEFAULT 0,
            feedback TEXT,
            logged_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS workout_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            plan TEXT,
            plan_start_date TEXT DEFAULT CURRENT_DATE,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    # ensure role and password_hash columns exist when migrating an existing database
    cur = db.execute("PRAGMA table_info(users)").fetchall()
    names = [r['name'] for r in cur]
    if 'role' not in names:
        db.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'")
    if 'password_hash' not in names:
        db.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
    # migrate workout_plans table
    wp_cur = db.execute("PRAGMA table_info(workout_plans)").fetchall()
    wp_names = [r['name'] for r in wp_cur]
    if 'plan_start_date' not in wp_names:
        db.execute("ALTER TABLE workout_plans ADD COLUMN plan_start_date TEXT")
        db.execute("UPDATE workout_plans SET plan_start_date = date('now') WHERE plan_start_date IS NULL")
    db.execute('''CREATE TABLE IF NOT EXISTS weight_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT,
        weight REAL, log_date TEXT, notes TEXT,
        logged_at TEXT DEFAULT CURRENT_TIMESTAMP)''')
    db.execute('''CREATE TABLE IF NOT EXISTS workout_completions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT,
        completion_date TEXT, completed INTEGER DEFAULT 0,
        workout_focus TEXT, duration INTEGER DEFAULT 0,
        notes TEXT, logged_at TEXT DEFAULT CURRENT_TIMESTAMP)''')
    db.execute('''CREATE TABLE IF NOT EXISTS diet_plans (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, plan TEXT,
        plan_start_date TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)''')
    db.execute('''CREATE TABLE IF NOT EXISTS meal_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, log_date TEXT,
        meal_index INTEGER, meal_name TEXT, completed INTEGER DEFAULT 0,
        feedback TEXT, logged_at TEXT DEFAULT CURRENT_TIMESTAMP)''')
    db.commit()
    db.close()

init_db()

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def get_or_create_user(username, role='user'):
    db = get_db()
    user = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
    if not user:
        uid = str(uuid.uuid4())
        db.execute('INSERT INTO users (id, username, role) VALUES (?, ?, ?)', (uid, username, role))
        db.execute('INSERT INTO gamification (user_id) VALUES (?)', (uid,))
        db.commit()
        user = db.execute('SELECT * FROM users WHERE id = ?', (uid,)).fetchone()
    else:
        # if existing user has different role, update it
        if user['role'] != role:
            db.execute('UPDATE users SET role=? WHERE id=?', (role, user['id']))
            db.commit()
            user = db.execute('SELECT * FROM users WHERE id = ?', (user['id'],)).fetchone()
    db.close()
    return dict(user)

def calculate_bmi(weight, height):
    """weight in kg, height in cm"""
    h = height / 100
    return round(weight / (h * h), 2)

def award_xp(user_id, xp, workout_type=None):
    db = get_db()
    db.execute('UPDATE gamification SET total_xp = total_xp + ? WHERE user_id = ?', (xp, user_id))

    # Level up logic (100 XP per level)
    gam = db.execute('SELECT total_xp FROM gamification WHERE user_id = ?', (user_id,)).fetchone()
    if gam:
        new_level = max(1, gam['total_xp'] // 100 + 1)
        db.execute('UPDATE gamification SET level = ? WHERE user_id = ?', (new_level, user_id))

    if workout_type:
        db.execute('UPDATE gamification SET total_workouts = total_workouts + 1 WHERE user_id = ?', (user_id,))
        db.execute('INSERT INTO workouts (user_id, workout_type, xp_earned) VALUES (?, ?, ?)',
                   (user_id, workout_type, xp))

    db.commit()
    db.close()

def check_and_award_badges(user_id):
    db = get_db()
    gam = db.execute('SELECT * FROM gamification WHERE user_id = ?', (user_id,)).fetchone()
    if not gam:
        db.close()
        return []
    
    badges = json.loads(gam['badges'])
    new_badges = []
    
    badge_rules = [
        ('first_workout', gam['total_workouts'] >= 1, '🏃 First Step'),
        ('week_warrior', gam['streak_days'] >= 7, '🔥 Week Warrior'),
        ('century', gam['total_xp'] >= 100, '💯 Century Club'),
        ('iron_will', gam['total_workouts'] >= 10, '💪 Iron Will'),
        ('elite', gam['total_xp'] >= 500, '⭐ Elite Athlete'),
    ]
    
    for badge_id, condition, name in badge_rules:
        if condition and badge_id not in badges:
            badges.append(badge_id)
            new_badges.append(name)
    
    db.execute('UPDATE gamification SET badges = ? WHERE user_id = ?', (json.dumps(badges), user_id))
    db.commit()
    db.commit()
    db.close()
    return new_badges

# ─── ROUTES ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/workout')
def workout():
    return render_template('workout.html')

@app.route('/diet')
def diet():
    return render_template('diet.html')

@app.route('/monitor')
def monitor():
    return render_template('monitor.html')

@app.route('/nutrition')
def nutrition():
    return render_template('nutrition.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/gamification')
def gamification():
    return render_template('gamification.html')

# ─── API: USER PROFILE ───────────────────────────────────────────────────────

@app.route('/api/user/setup', methods=['POST'])
def user_setup():
    data = request.get_json() or {}
    print(f"DEBUG: Received setup request for user: {data.get('username')}")
    
    username = data.get('username', 'user_' + str(uuid.uuid4())[:8])
    role = data.get('role', 'user')
    password = data.get('password')
    
    # All users now require passwords
    if not password:
        print(f"DEBUG: Setup failed - Password missing for {username}")
        return jsonify({'success': False, 'error': 'Password required'}), 400
    
    user = get_or_create_user(username, role)
    
    # If user is trainer, handle password specially
    db = get_db()
    try:
        if role == 'trainer':
            # Trainer: only username, role, and password
            from werkzeug.security import generate_password_hash
            pw_hash = generate_password_hash(password)
            db.execute('UPDATE users SET password_hash=?, role=? WHERE id=?',
                       (pw_hash, role, user['id']))
            db.commit()
            print(f"DEBUG: Trainer setup successful for {username} (ID: {user['id']})")
            return jsonify({'success': True, 'user_id': user['id']})
        
        from werkzeug.security import generate_password_hash
        pw_hash = generate_password_hash(password)
        bmi = calculate_bmi(float(data.get('weight', 70)), float(data.get('height', 170)))
        
        db.execute('''UPDATE users SET password_hash=?, age=?, gender=?, weight=?, height=?, bmi=?, role=?,
                      fitness_goal=?, activity_level=?, food_preference=?, target_weight=?, email=?
                      WHERE id=?''',
                   (pw_hash, data.get('age'), data.get('gender'), data.get('weight'), data.get('height'),
                    bmi, role, data.get('fitness_goal'), data.get('activity_level'),
                    data.get('food_preference'), data.get('target_weight'), data.get('email'), user['id']))
        db.commit()
        
        # Update streak
        today = str(date.today())
        gam = db.execute('SELECT last_login, streak_days FROM gamification WHERE user_id=?', (user['id'],)).fetchone()
        if gam:
            streak = gam['streak_days']
            if gam['last_login'] != today:
                streak += 1
            db.execute('UPDATE gamification SET last_login=?, streak_days=? WHERE user_id=?', (today, streak, user['id']))
            db.commit()
        
        print(f"DEBUG: User setup successful for {username} (ID: {user['id']})")
        return jsonify({'success': True, 'user_id': user['id'], 'bmi': bmi, 'username': username})
        
    except Exception as e:
        print(f"DEBUG: Setup error for {username}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        db.close()

@app.route('/api/user/<user_id>', methods=['GET', 'PATCH'])
def get_user(user_id):
    if request.method == 'PATCH':
        data = request.json or {}
        allowed = ['weight', 'height', 'age', 'target_weight', 'food_preference',
                   'fitness_goal', 'activity_level', 'gender']
        updates = {k: v for k, v in data.items() if k in allowed and v is not None}
        if updates:
            db = get_db()
            # Recalculate BMI if weight or height updated
            if 'weight' in updates or 'height' in updates:
                user = db.execute('SELECT weight, height FROM users WHERE id=?', (user_id,)).fetchone()
                w = float(updates.get('weight', user['weight'] or 70))
                h = float(updates.get('height', user['height'] or 170)) / 100
                updates['bmi'] = round(w / (h * h), 2)
            set_clause = ', '.join(f'{k}=?' for k in updates)
            db.execute(f'UPDATE users SET {set_clause} WHERE id=?', list(updates.values()) + [user_id])
            db.commit()
            db.close()
        return jsonify({'success': True})
    db = get_db()
    user = db.execute('SELECT * FROM users WHERE id=?', (user_id,)).fetchone()
    gam = db.execute('SELECT * FROM gamification WHERE user_id=?', (user_id,)).fetchone()
    db.close()
    if user:
        u = dict(user)
        u['gamification'] = dict(gam) if gam else {}
        return jsonify(u)
    return jsonify({'error': 'User not found'}), 404

# ─── API: AI WORKOUT RECOMMENDATION ─────────────────────────────────────────

@app.route('/api/workout/recommend', methods=['POST'])
def recommend_workout():
    data = request.json
    user_id = data.get('user_id')
    prompt = f"""
You are a professional fitness trainer. Create a detailed, personalized workout plan based on:

User Profile:
- Age: {data.get('age')} years
- Gender: {data.get('gender')}
- Weight: {data.get('weight')} kg
- Height: {data.get('height')} cm
- BMI: {data.get('bmi', 'N/A')}
- Fitness Goal: {data.get('fitness_goal')}
- Activity Level: {data.get('activity_level')}
- Available Equipment: {data.get('equipment', 'bodyweight only')}

Create a weekly workout plan with:
1. Daily schedule (which days to train, which to rest)
2. For each training day: exercise name, sets, reps/duration, rest time, intensity level
3. Warm-up and cool-down routines
4. Progression tips for the next 4 weeks

Format the response as structured JSON with this schema:
{{
  "plan_name": "...",
  "weekly_schedule": [
    {{
      "day": "Monday",
      "type": "training/rest",
      "focus": "...",
      "warmup": [...],
      "exercises": [
        {{"name": "...", "sets": 3, "reps": "10-12", "rest": "60s", "intensity": "moderate", "instructions": "..."}}
      ],
      "cooldown": [...]
    }}
  ],
  "tips": [...],
  "estimated_calories_per_session": 300
}}
"""
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_TEXT_MODEL,
            messages=[
                {'role': 'system', 'content': 'You are a certified fitness trainer and exercise physiologist. Return ONLY a valid JSON object matching the requested schema.'},
                {'role': 'user', 'content': prompt}
            ],
            response_format={'type': 'json_object'},
            max_tokens=3000
        )
        text = response.choices[0].message.content.strip()
        # Extract JSON from markdown code blocks if present
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()
        plan = json.loads(text)
        # persist plan if user_id provided
        if user_id:
            db = get_db()
            db.execute('INSERT INTO workout_plans (user_id, plan, plan_start_date) VALUES (?, ?, ?)',
                       (user_id, json.dumps(plan), str(date.today())))
            db.commit()
            db.close()
        return jsonify({'success': True, 'plan': plan})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ─── API: AI DIET RECOMMENDATION ─────────────────────────────────────────────

@app.route('/api/diet/recommend', methods=['POST'])
def recommend_diet():
    data = request.json
    prompt = f"""
You are a certified nutritionist. Create a detailed daily meal plan based on:

User Profile:
- Current Weight: {data.get('weight')} kg
- Target Weight: {data.get('target_weight')} kg
- Goal: {data.get('fitness_goal')} 
- Food Preference: {data.get('food_preference')}
- Activity Level: {data.get('activity_level')}
- Age: {data.get('age')}, Gender: {data.get('gender')}

Provide:
1. Daily calorie target
2. Macro split (protein/carbs/fats in grams and %)
3. Full day meal plan (breakfast, snack, lunch, snack, dinner)
4. Each meal: name, ingredients, calories, macros, prep time
5. Hydration recommendation
6. Supplement suggestions (optional)

Return as structured JSON:
{{
  "daily_calories": 2200,
  "macros": {{"protein": {{"grams": 165, "percent": 30}}, "carbs": {{"grams": 220, "percent": 40}}, "fats": {{"grams": 73, "percent": 30}}}},
  "meals": [
    {{
      "meal": "Breakfast",
      "time": "7:00 AM",
      "name": "...",
      "calories": 450,
      "protein": 35,
      "carbs": 45,
      "fats": 12,
      "ingredients": ["..."],
      "instructions": "..."
    }}
  ],
  "hydration": "...",
  "tips": [...]
}}
"""
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_TEXT_MODEL,
            messages=[
                {'role': 'system', 'content': 'You are a certified nutritionist. Return ONLY a valid JSON object matching the requested schema.'},
                {'role': 'user', 'content': prompt}
            ],
            response_format={'type': 'json_object'},
            max_tokens=3000
        )
        text = response.choices[0].message.content.strip()
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()
        plan = json.loads(text)
        return jsonify({'success': True, 'plan': plan})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ─── API: NUTRITION IMAGE RECOGNITION ────────────────────────────────────────

@app.route('/api/nutrition/analyze', methods=['POST'])
def analyze_food():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'})

    file = request.files['image']
    img_bytes = file.read()

    # convert image to base64 for vision API
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')

    # ── Step 1: Ask AI to identify the food item in the image ──────────
    try:
        detect_prompt = """
You are a world-class nutritional vision assistant. 
Analyze the attached image and identify the primary food item or meal.
BE EXTREMELY SPECIFIC and look for defining visual features:
- PIZZA: Look for distinct bread crust, melted cheese, tomato sauce, and toppings.
- SALAD: Look for leafy greens (lettuce, spinach), fresh vegetables, and protein chunks (like chicken).
- DIFFERENTIATION: Do not mistake a bowl of salad with chicken for a pizza just because of the colors. Check for the presence of a baked crust and dough.

Return ONLY a JSON object with this schema (no extra text, no markdown):
{
  "food_name": "exact name of the meal",
  "confidence": 0.0 to 1.0,
  "visual_description": "short description of what you see"
}
"""
        # Reset image file pointer
        file.seek(0)
        img_bytes = file.read()
        
        # Use content parts for better multi-modal performance
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        response = groq_client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[{'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{img_b64}'}},
                {'type': 'text', 'text': detect_prompt}
            ]}],
            max_tokens=512
        )
        text = response.choices[0].message.content.strip()
        # extract json if wrapped in code fences
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()
        detect_res = json.loads(text)
        display_name = detect_res.get('food_name', 'Unknown').replace('_', ' ').title()
    except Exception as e:
        # fallback to on-device model if API fails
        try:
            food_name, _confidence, _top5 = predict_food(img_bytes)
            display_name = food_name.replace('_', ' ').title()
        except Exception as e2:
            return jsonify({'success': False, 'error': f'Food detection failed: {e} / {e2}'})

    # ── Step 2: Get nutrition details for the detected food ──
    nutrition_prompt = f"""
You are a certified nutritionist. The food item detected is: \"{display_name}\".
Provide detailed nutritional information for a standard single serving of this food.
Return ONLY a valid JSON object with this exact schema (no markdown, no extra text):
{{
  "food_name": "{display_name}",
  "serving_size": "...",
  "calories": 0,
  "macros": {{
    "protein": 0,
    "carbs": 0,
    "fats": 0,
    "fiber": 0
  }},
  "micros": {{
    "vitamins": ["..."],
    "minerals": ["..."]
  }},
  "health_score": 0,
  "health_notes": "...",
  "alternatives": ["healthier alternative 1", "healthier alternative 2"]
}}
"""
    try:
        nutrition_response = groq_client.chat.completions.create(model=GROQ_TEXT_MODEL, messages=[{'role':'user','content':nutrition_prompt}], max_tokens=1024)
        text = nutrition_response.choices[0].message.content.strip()
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()
        result = json.loads(text)
        result['food_name'] = display_name  # ensure name matches model output
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        # If API fails, return detection name with a graceful fallback
        return jsonify({
            'success': True,
            'result': {
                'food_name': display_name,
                'serving_size': 'Standard serving',
                'calories': 0,
                'macros': {'protein': 0, 'carbs': 0, 'fats': 0, 'fiber': 0},
                'micros': {'vitamins': [], 'minerals': []},
                'health_score': 5,
                'health_notes': f'Nutrition data unavailable: {str(e)}',
                'alternatives': []
            }
        })

# ─── API: AI FITNESS CHATBOT ──────────────────────────────────────────────────

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    history = data.get('history', [])
    
    system_prompt = """You are FitAI, an expert AI fitness assistant. You help users with:
- Exercise techniques and form
- Workout planning and programming
- Nutrition and diet advice
- Recovery and injury prevention
- Motivation and fitness lifestyle tips
- Supplement guidance

Be encouraging, specific, and evidence-based. Keep responses concise but informative.
Use emojis occasionally to make responses engaging."""
    
    # Build proper Groq message format
    messages = [{'role': 'system', 'content': system_prompt}]
    if history:
        for h in history[-8:]:
            role = h.get('role', 'user')
            if role not in ('user', 'assistant'):
                role = 'user'
            messages.append({'role': role, 'content': h.get('content', '')})
    messages.append({'role': 'user', 'content': user_message})

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_TEXT_MODEL,
            messages=messages,
            max_tokens=1024
        )
        return jsonify({'success': True, 'response': response.choices[0].message.content})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ─── API: GAMIFICATION ────────────────────────────────────────────────────────

@app.route('/api/gamification/<user_id>', methods=['GET'])

@app.route('/api/workout/plan/<user_id>', methods=['GET'])
def get_workout_plan(user_id):
    db = get_db()
    row = db.execute('SELECT plan FROM workout_plans WHERE user_id=? ORDER BY created_at DESC LIMIT 1',
                     (user_id,)).fetchone()
    db.close()
    if row:
        try:
            plan = json.loads(row['plan'])
        except:
            plan = None
        return jsonify({'success': True, 'plan': plan})
    else:
        return jsonify({'success': False})


@app.route('/api/gamification/<user_id>', methods=['GET'])
def get_gamification(user_id):
    db = get_db()
    gam = db.execute('SELECT * FROM gamification WHERE user_id=?', (user_id,)).fetchone()
    if not gam:
        # Auto-create if missing
        db.execute('INSERT INTO gamification (user_id) VALUES (?)', (user_id,))
        db.commit()
        gam = db.execute('SELECT * FROM gamification WHERE user_id=?', (user_id,)).fetchone()
    workouts = db.execute('SELECT * FROM workouts WHERE user_id=? ORDER BY completed_at DESC LIMIT 10', (user_id,)).fetchall()
    leaderboard = db.execute('''SELECT u.username, g.total_xp, g.level, g.streak_days, g.total_workouts
                                 FROM gamification g JOIN users u ON g.user_id = u.id
                                 ORDER BY g.total_xp DESC LIMIT 10''').fetchall()
    db.close()
    
    if gam:
        badges = json.loads(gam['badges'])
        badge_info = {
            'first_workout': {'name': 'First Step', 'icon': '🏃', 'desc': 'Complete your first workout'},
            'week_warrior': {'name': 'Week Warrior', 'icon': '🔥', 'desc': '7-day streak'},
            'century': {'name': 'Century Club', 'icon': '💯', 'desc': 'Earn 100 XP'},
            'iron_will': {'name': 'Iron Will', 'icon': '💪', 'desc': 'Complete 10 workouts'},
            'elite': {'name': 'Elite Athlete', 'icon': '⭐', 'desc': 'Earn 500 XP'},
        }
        unlocked = [badge_info[b] for b in badges if b in badge_info]
        locked = [badge_info[b] for b in badge_info if b not in badges]
        
        return jsonify({
            'success': True,
            'gamification': dict(gam),
            'badges_unlocked': unlocked,
            'badges_locked': locked,
            'workouts': [dict(w) for w in workouts],
            'leaderboard': [dict(l) for l in leaderboard],
            'xp_to_next_level': 100 - (gam['total_xp'] % 100)
        })
    return jsonify({'success': False, 'error': 'User not found'})

@app.route('/api/gamification/log_workout', methods=['POST'])
def log_workout():
    data = request.json
    user_id = data.get('user_id')
    workout_type = data.get('workout_type', 'general')
    duration = data.get('duration', 30)
    
    # XP formula: 10 base + 1 per minute
    xp = 10 + duration
    award_xp(user_id, xp, workout_type)
    new_badges = check_and_award_badges(user_id)
    
    return jsonify({'success': True, 'xp_earned': xp, 'new_badges': new_badges})


# ─── AUTH & ADMIN ROUTES ───────────────────────────────────────────────────

@app.route('/logout')
def logout():
    """Clear any server-side session data and send the client back to the landing page.
    The front-end will wipe its own local storage state as well.
    """
    session.clear()
    from flask import redirect
    return redirect('/')


@app.route('/trainer')
def trainer_portal():
    """Simple trainer/admin interface that lists all users and their progress.
    Client-side JS will redirect non-trainers.
    """
    return render_template('trainer.html')


# ─── AUTHENTICATION ─────────────────────────────────────────────────────────
@app.route('/api/auth/login', methods=['POST'])
def auth_login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({'success': False, 'error': 'Username and password required'}), 400
    db = get_db()
    user = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
    db.close()
    if not user:
        return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
    from werkzeug.security import check_password_hash
    if not check_password_hash(user['password_hash'] or '', password):
        return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
    return jsonify({'success': True, 'user_id': user['id']})


@app.route('/api/trainer/users', methods=['GET'])
def api_trainer_users():
    # require the requester be a trainer
    requester_id = request.args.get('user_id')
    if not requester_id:
        return jsonify({'success': False, 'error': 'Missing user ID'}), 401
    db = get_db()
    r = db.execute('SELECT role FROM users WHERE id=?', (requester_id,)).fetchone()
    if not r or r['role'] != 'trainer':
        db.close()
        return jsonify({'success': False, 'error': 'Not authorized'}), 403

    rows = db.execute('''
        SELECT u.id, u.username, u.email, u.age, u.gender, u.weight, u.height, u.bmi,
               u.role, u.fitness_goal, u.activity_level, u.target_weight,
               g.total_xp, g.level, g.streak_days, g.total_workouts, g.last_login
        FROM users u
        LEFT JOIN gamification g ON u.id = g.user_id
        ORDER BY u.username
    ''').fetchall()
    users = []
    for r in rows:
        user = dict(r)
        # fetch latest plan if exists
        plan_row = db.execute('SELECT plan FROM workout_plans WHERE user_id=? ORDER BY created_at DESC LIMIT 1',
                              (user['id'],)).fetchone()
        if plan_row:
            try:
                user['latest_plan'] = json.loads(plan_row['plan'])
            except:
                user['latest_plan'] = None
        else:
            user['latest_plan'] = None
        users.append(user)
    db.close()
    return jsonify({'success': True, 'users': users})



# ─── API: TODAY'S WORKOUT ─────────────────────────────────────────────────────

@app.route('/api/workout/today/<user_id>', methods=['GET'])
def get_todays_workout(user_id):
    db = get_db()
    row = db.execute(
        'SELECT plan, plan_start_date FROM workout_plans WHERE user_id=? ORDER BY created_at DESC LIMIT 1',
        (user_id,)
    ).fetchone()
    db.close()

    if not row:
        return jsonify({'success': False, 'error': 'No workout plan found'})

    try:
        plan = json.loads(row['plan'])
        start_date_str = row['plan_start_date'] or str(date.today())
        start_date = date.fromisoformat(start_date_str)
        days_elapsed = (date.today() - start_date).days
        schedule = plan.get('weekly_schedule', [])
        if not schedule:
            return jsonify({'success': False, 'error': 'Plan has no schedule'})

        week_number = (days_elapsed // 7) + 1
        day_index = days_elapsed % 7
        # Wrap around if plan is complete
        day_index = day_index % len(schedule)
        todays_day = schedule[day_index]

        return jsonify({
            'success': True,
            'plan_name': plan.get('plan_name', 'Your Plan'),
            'week': week_number,
            'day_index': day_index + 1,
            'total_days': len(schedule),
            'start_date': start_date_str,
            'today': todays_day,
            'tips': plan.get('tips', []),
            'estimated_calories': plan.get('estimated_calories_per_session', 0)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/workout/reset_plan', methods=['POST'])
def reset_workout_plan():
    """Reset the plan start date so day tracking restarts from today."""
    data = request.json
    user_id = data.get('user_id')
    db = get_db()
    db.execute(
        'UPDATE workout_plans SET plan_start_date=? WHERE user_id=? AND id=(SELECT MAX(id) FROM workout_plans WHERE user_id=?)',
        (str(date.today()), user_id, user_id)
    )
    db.commit()
    db.close()
    return jsonify({'success': True})


# ─── API: SMART DIET (workout-aware) ─────────────────────────────────────────

@app.route('/api/diet/smart', methods=['POST'])
def smart_diet():
    """Generate a diet plan that accounts for today's workout intensity."""
    data = request.json
    user_id = data.get('user_id')

    # Pull user profile
    db = get_db()
    user = db.execute('SELECT * FROM users WHERE id=?', (user_id,)).fetchone()
    db.close()
    if not user:
        return jsonify({'success': False, 'error': 'User not found'})
    user = dict(user)

    # Pull today's workout
    today_res = get_todays_workout(user_id)  # reuse existing function logic
    today_data = today_res.get_json()

    workout_context = "rest day - light activity only"
    workout_name = "Rest Day"
    exercise_count = 0
    estimated_calories_burned = 0
    is_rest = True

    if today_data.get('success'):
        day = today_data.get('today', {})
        is_rest = day.get('type') == 'rest'
        workout_name = f"{day.get('day', '')} - {day.get('focus', '')}"
        exercise_count = len(day.get('exercises', []))
        estimated_calories_burned = today_data.get('estimated_calories', 300)
        if is_rest:
            workout_context = "rest/recovery day - minimal physical activity"
        else:
            workout_context = f"training day with {exercise_count} exercises focused on {day.get('focus', 'general fitness')}, estimated {estimated_calories_burned} calories burned"

    # Calculate TDEE
    weight = float(user.get('weight') or 70)
    height = float(user.get('height') or 170)
    age = int(user.get('age') or 25)
    gender = user.get('gender') or 'male'
    activity = user.get('activity_level') or 'moderately active'
    goal = user.get('fitness_goal') or 'maintenance'

    # Mifflin-St Jeor BMR
    if gender == 'female':
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age + 5

    activity_multipliers = {
        'sedentary': 1.2, 'lightly active': 1.375,
        'moderately active': 1.55, 'very active': 1.725, 'extremely active': 1.9
    }
    tdee = bmr * activity_multipliers.get(activity, 1.55)

    # Adjust for goal
    if 'loss' in goal: tdee -= 400
    elif 'gain' in goal or 'muscle' in goal: tdee += 300
    elif 'strength' in goal: tdee += 200

    # Adjust for today's workout
    if not is_rest:
        tdee += estimated_calories_burned * 0.5  # partial replenishment

    tdee = round(tdee)

    prompt = f"""You are a certified sports nutritionist. Create a precise daily meal plan for this athlete:

ATHLETE PROFILE:
- Weight: {weight}kg, Height: {height}cm, Age: {age}, Gender: {gender}
- Fitness Goal: {goal}
- Activity Level: {activity}
- Food Preference: {user.get('food_preference', 'non-vegetarian')}
- Target Weight: {user.get('target_weight', weight)}kg

TODAY'S WORKOUT: {workout_context}
Workout name: {workout_name}

CALCULATED TDEE: {tdee} calories

NUTRITION RULES based on today:
{"- REST DAY: Reduce carbs by 20%, maintain protein for muscle repair, moderate fats" if is_rest else f"- TRAINING DAY: Higher carbs for energy ({exercise_count} exercises), high protein for recovery, pre/post workout meals essential"}
- Protein target: {round(weight * (1.6 if not is_rest else 1.8))}g ({"recovery" if is_rest else "performance"} mode)
- Distribute calories across 5 meals timed around the workout

Return ONLY valid JSON with this exact schema:
{{
  "daily_calories": {tdee},
  "is_rest_day": {"true" if is_rest else "false"},
  "workout_context": "{workout_name}",
  "tdee_breakdown": {{
    "bmr": {round(bmr)},
    "activity_adjustment": {round(tdee - bmr)},
    "goal_adjustment": {round(-400 if 'loss' in goal else 300 if 'gain' in goal else 0)},
    "workout_bonus": {round(estimated_calories_burned * 0.5) if not is_rest else 0}
  }},
  "macros": {{
    "protein": {{"grams": 0, "percent": 0}},
    "carbs": {{"grams": 0, "percent": 0}},
    "fats": {{"grams": 0, "percent": 0}}
  }},
  "meals": [
    {{
      "meal": "Breakfast",
      "time": "7:00 AM",
      "purpose": "why this meal for today",
      "name": "meal name",
      "calories": 0,
      "protein": 0,
      "carbs": 0,
      "fats": 0,
      "ingredients": ["..."],
      "instructions": "brief prep instructions"
    }}
  ],
  "hydration": "specific hydration advice for today",
  "pre_workout": {"null" if is_rest else '"pre-workout snack recommendation"'},
  "post_workout": {"null" if is_rest else '"post-workout recovery meal tip"'},
  "tips": ["tip specific to today's workout/rest"]
}}"""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_TEXT_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=2048
        )
        text = response.choices[0].message.content.strip()
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()
        plan = json.loads(text)
        return jsonify({'success': True, 'plan': plan, 'tdee': tdee, 'is_rest': is_rest})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ─── API: USER TDEE ───────────────────────────────────────────────────────────

@app.route('/api/user/tdee/<user_id>', methods=['GET'])
def get_user_tdee(user_id):
    """Return calculated TDEE for the user to use as calorie goal."""
    db = get_db()
    user = db.execute('SELECT * FROM users WHERE id=?', (user_id,)).fetchone()
    db.close()
    if not user:
        return jsonify({'success': False})
    user = dict(user)
    weight = float(user.get('weight') or 70)
    height = float(user.get('height') or 170)
    age = int(user.get('age') or 25)
    gender = user.get('gender') or 'male'
    activity = user.get('activity_level') or 'moderately active'
    goal = user.get('fitness_goal') or 'maintenance'
    if gender == 'female':
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    multipliers = {'sedentary': 1.2, 'lightly active': 1.375, 'moderately active': 1.55, 'very active': 1.725, 'extremely active': 1.9}
    tdee = bmr * multipliers.get(activity, 1.55)
    if 'loss' in goal: tdee -= 400
    elif 'gain' in goal or 'muscle' in goal: tdee += 300
    return jsonify({'success': True, 'tdee': round(tdee), 'bmr': round(bmr)})


# ─── API: WEEKLY DIET PLAN (persist like workout plan) ───────────────────────

@app.route('/api/diet/generate_weekly', methods=['POST'])
def generate_weekly_diet():
    """Generate and persist a full 7-day diet plan linked to the workout plan."""
    data = request.json
    user_id = data.get('user_id')

    db = get_db()
    user = db.execute('SELECT * FROM users WHERE id=?', (user_id,)).fetchone()
    db.close()
    if not user:
        return jsonify({'success': False, 'error': 'User not found'})
    user = dict(user)

    # Pull workout plan for context
    db = get_db()
    wp_row = db.execute('SELECT plan FROM workout_plans WHERE user_id=? ORDER BY created_at DESC LIMIT 1', (user_id,)).fetchone()
    db.close()
    workout_schedule = []
    if wp_row:
        try:
            wp = json.loads(wp_row['plan'])
            workout_schedule = wp.get('weekly_schedule', [])
        except: pass

    weight = float(user.get('weight') or 70)
    height = float(user.get('height') or 170)
    age = int(user.get('age') or 25)
    gender = user.get('gender') or 'male'
    activity = user.get('activity_level') or 'moderately active'
    goal = user.get('fitness_goal') or 'maintenance'
    food_pref = user.get('food_preference') or 'non-vegetarian'

    if gender == 'female':
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    multipliers = {'sedentary': 1.2, 'lightly active': 1.375, 'moderately active': 1.55, 'very active': 1.725, 'extremely active': 1.9}
    tdee = round(bmr * multipliers.get(activity, 1.55))
    if 'loss' in goal: tdee -= 400
    elif 'gain' in goal or 'muscle' in goal: tdee += 300

    workout_context = ""
    if workout_schedule:
        workout_context = "\n".join([f"Day {i+1} ({d.get('day','')}) - {d.get('type','training')}: {d.get('focus','')} - {len(d.get('exercises',[]))} exercises" for i, d in enumerate(workout_schedule)])
    else:
        workout_context = "No specific workout plan - use general fitness guidelines"

    prompt = f"""You are a certified sports nutritionist. Create a 7-day meal plan. Return ONLY valid compact JSON, no extra text.

PROFILE: Weight {weight}kg, Height {height}cm, Age {age}, Gender {gender}, Goal: {goal}, Activity: {activity}, Diet: {food_pref}, TDEE: {tdee} kcal

WORKOUT SCHEDULE:
{workout_context}

RULES: Training days +200 kcal and higher carbs. Rest days lower carbs, higher protein. 4 meals per day only. Keep ingredient lists to max 3 items. Keep instructions under 10 words.

JSON schema (return exactly this structure for all 7 days):
{{"plan_name":"7-Day {goal.title()} Plan","base_tdee":{tdee},"weekly_days":[{{"day_number":1,"day_name":"Monday","workout_type":"training","workout_focus":"Upper Body","target_calories":2200,"calories_burned":300,"net_calories":1900,"macros":{{"protein":{{"grams":150}},"carbs":{{"grams":220}},"fats":{{"grams":70}}}},"meals":[{{"index":0,"meal":"Breakfast","time":"7:00 AM","name":"Oats with banana","calories":400,"protein":15,"carbs":60,"fats":8,"ingredients":["oats","banana","milk"],"workout_note":"Carbs for energy"}}],"hydration":"Drink 3L water","day_tip":"Focus on form"}}]}}"""

    def parse_json_safe(text):
        """Try to parse JSON, fixing truncation if needed."""
        text = text.strip()
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to fix truncated JSON by counting braces
            open_braces = text.count('{') - text.count('}')
            open_brackets = text.count('[') - text.count(']')
            if open_braces > 0 or open_brackets > 0:
                text = text + (']' * open_brackets) + ('}' * open_braces)
                try:
                    return json.loads(text)
                except:
                    pass
            raise

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_TEXT_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=8000
        )
        text = response.choices[0].message.content.strip()
        plan = parse_json_safe(text)

        # Persist the plan
        db = get_db()
        db.execute('INSERT INTO diet_plans (user_id, plan, plan_start_date) VALUES (?,?,?)',
                   (user_id, json.dumps(plan), str(date.today())))
        db.commit()
        db.close()
        return jsonify({'success': True, 'plan': plan})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/diet/plan/<user_id>', methods=['GET'])
def get_diet_plan(user_id):
    """Get the latest saved diet plan."""
    db = get_db()
    row = db.execute('SELECT plan, plan_start_date FROM diet_plans WHERE user_id=? ORDER BY created_at DESC LIMIT 1', (user_id,)).fetchone()
    db.close()
    if not row:
        return jsonify({'success': False})
    try:
        plan = json.loads(row['plan'])
        return jsonify({'success': True, 'plan': plan, 'start_date': row['plan_start_date']})
    except:
        return jsonify({'success': False})


@app.route('/api/diet/today/<user_id>', methods=['GET'])
def get_todays_diet(user_id):
    """Get today's diet day from the saved plan."""
    db = get_db()
    row = db.execute('SELECT plan, plan_start_date FROM diet_plans WHERE user_id=? ORDER BY created_at DESC LIMIT 1', (user_id,)).fetchone()
    db.close()
    if not row:
        return jsonify({'success': False, 'error': 'No diet plan found'})
    try:
        plan = json.loads(row['plan'])
        start_date = date.fromisoformat(row['plan_start_date'] or str(date.today()))
        days_elapsed = (date.today() - start_date).days
        days = plan.get('weekly_days', [])
        if not days:
            return jsonify({'success': False, 'error': 'Plan has no days'})
        day_index = days_elapsed % len(days)
        today_day = days[day_index]

        # Load meal completion logs for today
        db = get_db()
        logs = db.execute(
            'SELECT meal_index, completed, feedback FROM meal_logs WHERE user_id=? AND log_date=?',
            (user_id, str(date.today()))
        ).fetchall()
        db.close()
        meal_status = {r['meal_index']: {'completed': bool(r['completed']), 'feedback': r['feedback']} for r in logs}

        return jsonify({
            'success': True,
            'plan_name': plan.get('plan_name', ''),
            'base_tdee': plan.get('base_tdee', 2000),
            'day_number': day_index + 1,
            'total_days': len(days),
            'today': today_day,
            'meal_status': meal_status,
            'start_date': str(start_date)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/diet/meal_log', methods=['POST'])
def log_meal():
    """Mark a meal as completed and optionally add feedback."""
    data = request.json
    user_id = data.get('user_id')
    meal_index = data.get('meal_index')
    meal_name = data.get('meal_name', '')
    completed = 1 if data.get('completed', True) else 0
    feedback = data.get('feedback', '')
    today = str(date.today())

    db = get_db()
    existing = db.execute(
        'SELECT id FROM meal_logs WHERE user_id=? AND log_date=? AND meal_index=?',
        (user_id, today, meal_index)
    ).fetchone()
    if existing:
        db.execute('UPDATE meal_logs SET completed=?, feedback=?, logged_at=CURRENT_TIMESTAMP WHERE id=?',
                   (completed, feedback, existing['id']))
    else:
        db.execute('INSERT INTO meal_logs (user_id, log_date, meal_index, meal_name, completed, feedback) VALUES (?,?,?,?,?,?)',
                   (user_id, today, meal_index, meal_name, completed, feedback))
    db.commit()

    # Award XP for completing meals
    if completed:
        db.execute('UPDATE gamification SET total_xp = total_xp + 5 WHERE user_id=?', (user_id,))
        db.commit()
    db.close()
    return jsonify({'success': True, 'xp_earned': 5 if completed else 0})


@app.route('/api/diet/meal_logs/<user_id>', methods=['GET'])
def get_meal_logs(user_id):
    """Get all meal logs for today."""
    today = str(date.today())
    db = get_db()
    logs = db.execute(
        'SELECT * FROM meal_logs WHERE user_id=? AND log_date=?', (user_id, today)
    ).fetchall()
    db.close()
    return jsonify({'success': True, 'logs': [dict(l) for l in logs]})


# ─── API: WEIGHT TRACKING ─────────────────────────────────────────────────────

@app.route('/api/weight/log', methods=['POST'])
def log_weight():
    data = request.json
    user_id = data.get('user_id')
    weight = float(data.get('weight', 0))
    notes = data.get('notes', '')
    today = str(date.today())

    if not weight or weight <= 0:
        return jsonify({'success': False, 'error': 'Invalid weight'})

    db = get_db()
    # Upsert today's weight
    existing = db.execute('SELECT id FROM weight_logs WHERE user_id=? AND log_date=?', (user_id, today)).fetchone()
    if existing:
        db.execute('UPDATE weight_logs SET weight=?, notes=? WHERE id=?', (weight, notes, existing['id']))
    else:
        db.execute('INSERT INTO weight_logs (user_id, weight, log_date, notes) VALUES (?,?,?,?)',
                   (user_id, weight, today, notes))

    # Update user current weight
    db.execute('UPDATE users SET weight=?, bmi=? WHERE id=?',
               (weight, round(weight / ((db.execute("SELECT height FROM users WHERE id=?", (user_id,)).fetchone()['height'] / 100) ** 2), 2), user_id))
    db.commit()
    db.close()
    return jsonify({'success': True})


@app.route('/api/weight/history/<user_id>', methods=['GET'])
def get_weight_history(user_id):
    db = get_db()
    logs = db.execute(
        'SELECT weight, log_date, notes FROM weight_logs WHERE user_id=? ORDER BY log_date DESC LIMIT 30',
        (user_id,)
    ).fetchall()
    user = db.execute('SELECT weight, target_weight, height FROM users WHERE id=?', (user_id,)).fetchone()
    db.close()

    logs_list = [dict(l) for l in logs]

    # Calculate progress
    progress = {}
    if user and logs_list:
        current = float(user['weight'] or 0)
        target = float(user['target_weight'] or current)
        start = float(logs_list[-1]['weight']) if logs_list else current
        total_to_lose = abs(start - target)
        lost_so_far = abs(start - current)
        progress = {
            'current_weight': current,
            'target_weight': target,
            'start_weight': start,
            'change': round(current - start, 1),
            'remaining': round(abs(current - target), 1),
            'percent': round((lost_so_far / total_to_lose * 100) if total_to_lose > 0 else 100, 1),
            'on_track': (current <= target if target < start else current >= target)
        }

    return jsonify({'success': True, 'logs': logs_list, 'progress': progress})


# ─── API: WORKOUT COMPLETION ──────────────────────────────────────────────────

@app.route('/api/workout/complete', methods=['POST'])
def complete_workout():
    data = request.json
    user_id = data.get('user_id')
    completed = 1 if data.get('completed', True) else 0
    workout_focus = data.get('workout_focus', '')
    duration = int(data.get('duration', 45))
    notes = data.get('notes', '')
    today = str(date.today())

    db = get_db()
    existing = db.execute(
        'SELECT id FROM workout_completions WHERE user_id=? AND completion_date=?', (user_id, today)
    ).fetchone()
    if existing:
        db.execute('UPDATE workout_completions SET completed=?, duration=?, notes=?, workout_focus=? WHERE id=?',
                   (completed, duration, notes, workout_focus, existing['id']))
    else:
        db.execute('INSERT INTO workout_completions (user_id, completion_date, completed, workout_focus, duration, notes) VALUES (?,?,?,?,?,?)',
                   (user_id, today, completed, workout_focus, duration, notes))

    xp_earned = 0
    if completed:
        xp_earned = 10 + duration
        db.execute('UPDATE gamification SET total_xp=total_xp+?, total_workouts=total_workouts+1 WHERE user_id=?',
                   (xp_earned, user_id))
    db.commit()
    db.close()
    return jsonify({'success': True, 'xp_earned': xp_earned})


@app.route('/api/workout/completion_status/<user_id>', methods=['GET'])
def get_workout_completion(user_id):
    today = str(date.today())
    db = get_db()
    row = db.execute(
        'SELECT * FROM workout_completions WHERE user_id=? AND completion_date=?', (user_id, today)
    ).fetchone()
    # Last 7 days completion rate
    week_rows = db.execute(
        'SELECT completed FROM workout_completions WHERE user_id=? ORDER BY completion_date DESC LIMIT 7',
        (user_id,)
    ).fetchall()
    db.close()
    completion_rate = 0
    if week_rows:
        completion_rate = round(sum(1 for r in week_rows if r['completed']) / len(week_rows) * 100)
    return jsonify({
        'success': True,
        'today': dict(row) if row else None,
        'completion_rate': completion_rate
    })


# ─── API: DAILY SUMMARY ───────────────────────────────────────────────────────

@app.route('/api/summary/<user_id>', methods=['GET'])
def daily_summary(user_id):
    """Combined daily check-in data: weight, meals, workout."""
    today = str(date.today())
    db = get_db()

    # Today's weight log
    weight_row = db.execute('SELECT weight FROM weight_logs WHERE user_id=? AND log_date=?', (user_id, today)).fetchone()

    # Today's meal completion
    meal_logs = db.execute('SELECT completed FROM meal_logs WHERE user_id=? AND log_date=?', (user_id, today)).fetchall()
    meals_completed = sum(1 for m in meal_logs if m['completed'])
    meals_total = len(meal_logs)

    # Today's workout
    workout_row = db.execute('SELECT completed, workout_focus FROM workout_completions WHERE user_id=? AND completion_date=?',
                             (user_id, today)).fetchone()

    # User target
    user = db.execute('SELECT weight, target_weight FROM users WHERE id=?', (user_id,)).fetchone()
    db.close()

    return jsonify({
        'success': True,
        'today': today,
        'weight_logged': weight_row is not None,
        'current_weight': weight_row['weight'] if weight_row else (user['weight'] if user else None),
        'target_weight': user['target_weight'] if user else None,
        'meals_completed': meals_completed,
        'meals_total': meals_total,
        'workout_completed': bool(workout_row and workout_row['completed']) if workout_row else False,
        'workout_focus': workout_row['workout_focus'] if workout_row else None
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
