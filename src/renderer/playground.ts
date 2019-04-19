import console = require("console");

console.log("Playground!!")

import fs = require('fs');
function readFileAsync(filename: string): Promise<any> {
   return new Promise((resolve,reject) => {
       fs.readFile(filename,(err,result) => {
           if (err) reject(err);
           else resolve(result);
       });
   });
}

const delay = (ms: number) => new Promise(res => setTimeout(res, ms));


// delay(1000).then(result =>{console.log(result)})

function iReturnPromiseAfter1Second(numTimeOut): Promise<string> {
    return new Promise((resolve) => {
        setTimeout(() => resolve("Hello world!"), numTimeOut);
    });
 }

 // async before a function, run sequences of promises 
 async function sequentialFunction(){

    // step Z, grab JSON

    // turning the callback into a promise  
    const promise=new Promise((resolve,reject) => {
            fs.readFile("state.json",(err,result) => {
            if (err) reject(err);
            else resolve(result);
            })});
    // promise.then(result=>JSON.parse(result));
    // const promiseVal=await promise();
    
    //   );

    // console.log(metadataToHighlight)

    //step A, foo() 
    try {
        var val =  iReturnPromiseAfter1Second(1000);//only use await in a function that is async function 
        var val2 =  iReturnPromiseAfter1Second(1000);
        var val3 =  iReturnPromiseAfter1Second(1000);
        console.log("once",val);  // then(result =>{console.log(result)}); 
        Promise.all([val, val2, val3]).then(result=>{}) // only return when they are all done. 
        console.log("twice",val)
    }
    catch(err) {
        console.log('Error: ', err.message);
    }
     

    //step B, then().then().then() nested 
    iReturnPromiseAfter1Second(1000).then(result =>{console.log("returned after happyreturn"+result)})
    return "Happy return"
    //return early and keep iReturnPromise going?
 }

 const result=sequentialFunction()
 console.log(result)



