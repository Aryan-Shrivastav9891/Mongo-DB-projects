const express = require("express")
const {MongoClient , ServerApiVersion} = require("mongodb")
const EmployeeRoute = require("./routes/addRouter.js")
const env = require("dotenv").config()
const app = express()
const port = process.env.PORT || 8082

console.log(port);

app.get("/" , (req ,res)=>{
    res.send("sucessfully")
})

app.use ("/api/employee" , EmployeeRoute)

app.listen(port , ()=>{
    console.log(`server connnected ${port}`);
})
