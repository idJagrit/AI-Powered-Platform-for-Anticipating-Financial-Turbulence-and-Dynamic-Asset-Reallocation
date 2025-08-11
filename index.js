import express from "express";
import bodyParser from "body-parser";
import pg from "pg";
import path from "path";
import '../styles/analysis.css';
import { fileURLToPath } from "url";
import dotenv from "dotenv";
dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const db = new pg.Client({
  user: process.env.DB_USER,
  host: process.env.DB_HOST,
  database: process.env.DB_NAME,
  password: process.env.DB_PASSWORD,
  port: process.env.DB_PORT,
  ssl: {
    rejectUnauthorized: false,
  }
});

db.connect();

const app = express();

app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static("public"));
app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "views"));

app.get("/", (req, res) => {
  res.render("index");
});

app.get("/signup", (req, res) => {
  res.render("signup");
});

app.post("/signup", async (req, res) => {
  const { username, password } = req.body;
  try {
    const checkUser = await db.query("SELECT * FROM users WHERE username = $1", [username]);
    if (checkUser.rows.length > 0) {
      res.render("signup", { error: "Username already exists. Try signing in." });
    } else {
      await db.query("INSERT INTO users (username, password) VALUES ($1, $2)", [username, password]);
      res.render("dashboard", { username });
    }
  } catch (err) {
    console.error(err);
    res.status(500).send("Database error during signup.");
  }
});

app.get("/signin", (req, res) => {
  res.render("signin");
});

app.post("/signin", async (req, res) => {
  const { username, password } = req.body;

  try {
    const result = await db.query("SELECT * FROM users WHERE username = $1 AND password = $2", [username, password]);
    if (result.rows.length > 0) {
      res.render("dashboard", { username });
    } else {
      res.render("signin", { error: "Incorrect username or password." });
    }
  } catch (err) {
    console.error(err);
    res.status(500).send("Database error during signin.");
  }
});

app.get("/dashboard", (req, res) => {
  res.render("dashboard");
});

app.get('/analysis', (req, res) => {
  const username = req.query.username;
  res.render('analysis', { username });
});

export default app;

