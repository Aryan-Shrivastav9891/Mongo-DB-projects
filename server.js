const {MongoClient, ServerApiVersion} = require("mongodb");
const express = require("express");
const app = express();
const PORT = 8082;

// Encode the special character '@' in the password using %40
const mongoDB = "mongodb+srv://WizAryan:Aryan%40111@cluster0.vlrjc.mongodb.net/crud_pro_1";

const DBclient = new MongoClient(mongoDB, {
    serverApi: {
        version: ServerApiVersion.v1,
        strict: true,
        deprecationErrors: true,
    },
});

app.use(express.json());

const dbConnection = async () => {
    try {
        await DBclient.connect();
        console.log("MongoDB connected");
    } catch (error) {
        console.log(error, "error");
    }
};
dbConnection();

app.get("/", (req, res) => {
    res.send("Connection successful");
});

app.get("/getAllData", async (req, res) => {
    try {
        const database = DBclient.db("Students");
        const students = database.collection("students");
        const result = await students.find({}).toArray()
        if (!result) {
            res.json({
                status:"false",
                message:"data not present"
            })
        }else{
            res.json({
                status:"true",
                message:"data is hear",
                data : result
            })
        }
        res.json({
            status:"true",
            message:""
        })

    } catch (error) {
        console.log(error, "this is an error in MongoDB");
        res.status(500).json({
            status: 500,
            message: "Error fetching data",
            error: error.message,
        });
    }
});
app.post("/add_Data", async (req, res) => {
    try {
        const database = DBclient.db("Students");
        const students = database.collection("students");
        // Assuming the data is sent in the request body
        const datastu = req.body;
        if (!datastu["name"]) {
            res.json({
                status: 200,
                message: "data not find",
            });
        } else {
            const result = await students.insertOne(datastu);
            console.log(result);
            res.json({
                status: 200,
                message: "data send sucessfully",
            });
        }
    } catch (error) {
        console.log(error, "this is an error in MongoDB");
    }
});

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
