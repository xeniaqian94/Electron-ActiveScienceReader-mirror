import fs = require("fs-extra");
import os = require("os");
import path = require("path");
import { listPdfs } from "../renderer/io";
var iconv = require('iconv-lite');

const doIt = async () => {
    const pdfRootDir = 'C:\\Users\\merha\\Desktop'
    const pdfPathInfo = await listPdfs(pdfRootDir)
    let neededDir = pdfPathInfo[0].fileNameWithExt
    .replace(/\.pdf/, "")
    .replace(/\s/g, "-")
    .replace(/%/g, "-");
    
    // console.log('asdf', str, neededDir)
    await fs.ensureDir(path.join(pdfRootDir, neededDir));
    await fs.move(
      pdfPathInfo[0].fullFilePath,
      path.join(path.join(pdfRootDir, neededDir), "./" + neededDir + ".pdf")
    );
}
doIt()

// import { dispatch, getState } from "./createStore";
// import { makePdfSegmentViewbox } from "./creators";
// import { NestedPartial } from "../renderer/utils";

// // console.log("init", getState());

// const vbs = [...Array(10)].map(x => makePdfSegmentViewbox());
// const vb = makePdfSegmentViewbox();
// const vb2 = makePdfSegmentViewbox();

// // dispatch.graph.addBatch({
// //   nodes: vbs.slice(0, 2)
// // });
// console.time('add batch')
// dispatch.graph.addBatch({nodes: [vbs]})
// dispatch.graph.toggleSelections({selectedNodes: [vbs[0].id]})

// console.timeEnd('add batch')

// const updates = vbs
//   .slice(0, 300)
//   .map(x => ({ id: x.id, data: { left: 3, top: 10 } }));
// console.time('time')

// const nodeUpdate = {
//   id: vbs[0].id,
//   data: {height: 1, }
// } 
// const nodeUpdate2 = {
//     id: vbs[1].id,
//     data: {height: 222, }
//   }
// dispatch.graph.updateBatch( {nodes: [nodeUpdate]});
// console.dir(getState().graph.nodes[vbs[1].id]);
// console.timeEnd('time')

// dispatch.app.deleteNodes([vb.id])
// console.dir(getState().app.nodes);
