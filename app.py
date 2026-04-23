from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
import pandas as pd
import numpy as np
import io
import os
from functools import wraps

app = Flask(__name__)

# CORS Configuration
CORS(app, 
     origins=["https://data-cleaning-website.vercel.app"],
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "OPTIONS"],
     supports_credentials=True)

# CONFIG
app.config["MONGO_URI"] = os.environ.get("MONGO_URI", "mongodb://localhost:27017/datacleaner")
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "your_secret_key_change_this")
mongo = PyMongo(app)
user_dfs = {}


# JWT DECORATOR
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if not token:
            return jsonify({"error": "Token missing"}), 401
        try:
            data = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except Exception:
            return jsonify({"error": "Invalid token"}), 401
        return f(data["user_id"], *args, **kwargs)
    return decorated

# AUTH ROUTES
@app.route("/api/signup", methods=["POST", "OPTIONS"])
def signup():
    if request.method == "OPTIONS":
        return make_response(), 200
    data = request.get_json()
    name     = data.get("name", "").strip()
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")
    if not name or not email or not password:
        return jsonify({"error": "All fields required"}), 400
    if mongo.db.users.find_one({"email": email}):
        return jsonify({"error": "Email already registered"}), 409
    hashed = generate_password_hash(password)
    mongo.db.users.insert_one({"name": name, "email": email, "password": hashed})
    return jsonify({"message": "Account created successfully"}), 201

@app.route("/api/login", methods=["POST", "OPTIONS"])
def login():
    if request.method == "OPTIONS":
        return make_response(), 200
    data     = request.get_json()
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")
    user = mongo.db.users.find_one({"email": email})
    if not user or not check_password_hash(user["password"], password):
        return jsonify({"error": "Invalid email or password"}), 401
    token = jwt.encode(
        {"user_id": str(user["_id"]), "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)},
        app.config["SECRET_KEY"], algorithm="HS256",
    )
    return jsonify({"token": token, "name": user["name"]}), 200

# FILE UPLOAD
@app.route("/api/file", methods=["POST", "OPTIONS"])
@token_required
def upload_file(user_id):
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files["file"]
        fname = file.filename.lower()
        if fname.endswith(".csv"):
            df = pd.read_csv(file)
        elif fname.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Only CSV and Excel files allowed"}), 400
        user_dfs[user_id] = df
        col_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            if "int" in dtype or "float" in dtype:
                suggested = "numeric"
            elif "datetime" in dtype:
                suggested = "date"
            else:
                suggested = "categorical"
            col_info.append({"name": col, "suggested": suggested})
        return jsonify({"cols": col_info, "rows": len(df)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# CLEAN & DOWNLOAD
@app.route("/api/inputs", methods=["POST", "OPTIONS"])
@token_required
def form_input(user_id):
    try:
        df = user_dfs.get(user_id)
        if df is None:
            return jsonify({"error": "No file uploaded. Please upload a file first."}), 400
        payload = request.get_json()
        if not payload:
            return jsonify({"error": "No configuration received"}), 400
        options = {}
        for col, cfg in payload.items():
            dtype = cfg.get("type", "ignore")
            if dtype == "ignore":
                continue
            options[col] = [dtype]
            if dtype == "numeric":
                options[col].append([float(cfg["min"]), float(cfg["max"])])
                options[col].append(cfg["handle"])
            elif dtype == "categorical":
                cats = [c.strip() for c in cfg["cats"].split(",") if c.strip()]
                options[col].append(cats)
                options[col].append(cfg["handle"])
            elif dtype == "date":
                options[col].append([])
                options[col].append(cfg["handle"])
        cleaned_df = clean_dataset(options, df.copy())
        output = io.StringIO()
        cleaned_df.to_csv(output, index=False)
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode("utf-8")),
            mimetype="text/csv",
            as_attachment=True,
            download_name="cleaned_file.csv",
        )
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

def clean_dataset(options, df):
    for col in df.columns:
        if col not in options:
            continue
        dtype, *rest = options[col]
        if dtype == "numeric":
            bounds, handle = rest
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[(df[col] < bounds[0]) | (df[col] > bounds[1]), col] = np.nan
            if handle == "remove":
                df = df[df[col].notna()]
            elif handle == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif handle == "median":
                df[col] = df[col].fillna(df[col].median())
            elif handle == "mode":
                df[col] = df[col].fillna(df[col].mode()[0])
        elif dtype == "categorical":
            cats, handle = rest
            df[col] = df[col].astype(str).str.lower().str.strip()
            cats = [c.lower() for c in cats]
            df[col] = pd.Categorical(df[col], categories=cats)
            if handle == "remove":
                df = df[df[col].notna()]
            elif handle == "mode":
                df[col] = df[col].fillna(df[col].mode()[0])
        elif dtype == "date":
            _, handle = rest
            df[col] = pd.to_datetime(df[col], errors="coerce")
            if handle == "remove":
                df = df[df[col].notna()]
            elif handle == "mode":
                df[col] = df[col].fillna(df[col].mode()[0])
    df = df.drop_duplicates().reset_index(drop=True)
    return df

if __name__ == "__main__":
    app.run(debug=True)