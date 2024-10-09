const express = require("express");
const app = express();
const path = require("path");
require("dotenv").config();
const ejsLayout = require("express-ejs-layouts");

const port = process.env.PORT || 3000;
const { MongoClient, ServerApiVersion } = require("mongodb");
const MongoUrl = process.env.url;

app.set("view engine", "ejs");
app.use(ejsLayout);
app.set("layout", "layout/common");
app.use(express.urlencoded({ extended: true }));

let NewStd , database

const Client = new MongoClient(MongoUrl, {
  serverApi: {
    version: ServerApiVersion.v1,
    strict: true,
    deprecationErrors: true,
  },
});

const connectDB = async () => {
  try {
    await Client.connect();
    console.log("mongoDb connected true✅");
  } catch (err) {
    console.log(err);
  }
};
connectDB();

app.use(express.json());
app.use(express.static(path.join(__dirname, "views")));

app.get("/", (req, res) => {
  res.render("home");
});
app.get("/List", (req, res) => {
  const data = NewStd.find().toArray()
  res.render("listData" , {data});
});

//! add data
app.get("/add", (req, res) => {
  res.render("addData");
});

app.post("/add", async (req, res) => {
  const Student = {
    name: req.body.name,
    age: parseInt(req.body.age),
    email: req.body.email,
  };
  console.log(Student);
  // res.render("submit");

  try {
    database = Client.db("NewStudent");
    NewStd = database.collection("NewStd");
    const result = await NewStd.insertOne(Student);
    console.log(result);
    res.render("submit", { student: Student });
  } catch (err) {
    console.log(err);
  }
});

app.listen(port, () => {
  console.log(`server is running on port ${port}`);
});
