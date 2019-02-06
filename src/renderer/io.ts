import fs = require("fs-extra");
import os = require("os");
import path = require("path");
import glob = require("glob");
import uuidv1 = require("uuid/v1");
import jsonfile = require("jsonfile");
import * as _pdfjs from "pdfjs-dist";
var PdfjsWorker = require("pdfjs-dist/lib/pdf.worker.js");
if (typeof window !== "undefined" && "Worker" in window) {
  (_pdfjs as any).GlobalWorkerOptions.workerPort = new PdfjsWorker();
}
import {
  PDFJSStatic,
  PDFDocumentProxy,
  PDFInfo,
  PDFMetadata,
  PDFTreeNode
} from "pdfjs-dist";
const pdfjs: PDFJSStatic = _pdfjs as any;

import {
  flatten,
  zeroPad,
  unique,
  brewer12,
  roundedToFixed,
  sortBy
} from "./utils";

import { histogram, mean, median, deviation } from "d3-array";

interface FileInfo {
  pdfPath: string;
  pdfName: string;
  dataDir: string;
}

export const listPdfs = (fullPath: string): Promise<FileInfo[]> =>
  new Promise(resolve => {
    glob(fullPath + "/*.pdf", { nodir: true }, (err, files) => {
      const fileInfo = files.map(f => {
        return {
          pdfPath: f,
          pdfName: path.basename(f),
          dataDir: f.replace(".pdf", "")
        };
      });
      resolve(fileInfo);
    });
  });

export const listDirs = (fullpath: string): Promise<string[]> => {
  return new Promise(resolve => {
    glob(fullpath + "/*/", {}, (err, files) => {
      resolve(files);
    });
  });
};

export const ls = (fullpath: string, options = {}): Promise<string[]> => {
  return new Promise(resolve => {
    glob(fullpath, options, (err, files) => {
      resolve(files);
    });
  });
};

export const setupDirFromPdfs = async () => {
  const { homedir, username } = os.userInfo();
  const pdfDir = path.join(homedir, "pdfs");
  const infos = await listPdfs(pdfDir);

  for (let info of infos) {
    const dirExists = fs.pathExists(info.dataDir);
    const postFix = dirExists ? "_" + uuidv1() : "";
    const dataDir = dirExists ? info.dataDir + postFix : info.dataDir;
    await fs.ensureDir(dataDir);
    await fs.move(
      info.pdfPath,
      path.join(dataDir, info.pdfName.replace(".pdf", postFix + ".pdf"))
    );
  }

  const dirs = await listDirs(pdfDir);
  await preprocessPdfs(dirs)();
};

setupDirFromPdfs();

export const existsElseMake = async (
  path: string,
  promise: _pdfjs.PDFPromise<any> | Promise<any> | {},
  overwrite = false
) => {
  const fileExists = await fs.pathExists(path);

  if (!fileExists || overwrite) {
    console.log("making ", path);
    const data = await promise;
    await jsonfile.writeFile(path, data);
  }
};

export const preprocessPdfs = (
  pdfDirs: string[],
  overwrite = false
) => async () => {
  console.log("preprocessing pdf ", pdfDirs);
  // console.time("time");
  const scale = 1;
  //   const dir = pdfDirs[0];
  for (let dir of pdfDirs) {
    const files = await ls(dir + "/*");
    const [pdfPath] = files.filter(x => x.endsWith(".pdf"));
    const pdf = await pdfjs.getDocument(pdfPath);
    const allPageNumbers = [...Array(pdf.numPages).keys()].map(x => x + 1);

    await existsElseMake(
      path.join(dir, "meta.json"),
      pdf.getMetadata(),
      overwrite
    );
    await existsElseMake(
      path.join(dir, "outline.json"),
      pdf.getOutline(),
      overwrite
    );

    for (const pageNumber of allPageNumbers) {
      const pageId = zeroPad(pageNumber, 4);
      // console.log(pageNumber)

      const textToDisplayFile = path.join(
        dir,
        `textToDisplay-page${pageId}.json`
      );

      const fileExists = await fs.pathExists(textToDisplayFile);

      if (!fileExists || overwrite) {
        const page = await pdf.getPage(pageNumber);
        const viewport = page.getViewport(scale);
        const text = await page.getTextContent();

        const [xMin, yMin, xMax, yMax] = (viewport as any).viewBox;
        const { width, height } = viewport;
        const textToDisplay = await Promise.all(
          text.items.map(async (tc, i) => {
            const fontData = await (page as any).commonObjs.ensureObj(
              tc.fontName
            );

            const [, , , offsetY, x, y] = tc.transform;
            const top = yMax - (y + offsetY);
            const left = x - xMin;
            const fontHeight = tc.transform[3];

            return {
              ...tc,
              id: pageId + "-" + zeroPad(i, 4),
              top: top * scale,
              left: left * scale,
              width: tc.width * scale,
              height: tc.height * scale,
              fontHeight: tc.transform[3],
              fontWidth: tc.transform[0],
              scaleX: tc.transform[0] / tc.transform[3],
              fallbackFontName: fontData.data
                ? fontData.data.fallbackName
                : "sans-serif",
              style: text.styles[tc.fontName]
            };
          })
        );

        await jsonfile.writeFile(textToDisplayFile, {
          pageNumber,
          text: textToDisplay,
          // maybe use this to detect rotation?
          viewportFlat: { width, height, xMin, yMin, xMax, yMax }
        });
        console.log("making ", textToDisplayFile);
      }
    }
    // use this to know if processing is done
    await existsElseMake(path.join(dir, "textToDisplay.json"), {
      numberOfPages: pdf.numPages,
      createdOn: new Date()
    });

    let pagesOfText = await loadPageJson(dir, "textToDisplay");
    const columnLefts = getLeftEdgeOfColumns(pagesOfText);
    await existsElseMake(path.join(dir, `columnLefts.json`), columnLefts);

    for (let page of pagesOfText) {
      const pageId = zeroPad(page.pageNumber, 4);
      await existsElseMake(
        path.join(dir, `linesOfText-page${pageId}.json`),
        getLines(columnLefts, page.text, page.pageNumber)
      );
    }

    // use this to know if processing is done
    await existsElseMake(path.join(dir, "linesOfText.json"), {
      numberOfPages: pdf.numPages,
      createdOn: new Date()
    });
  }
  // console.timeEnd("time");
};

export const fontStats = (pages: PageToDisplay[]) => {
  // todo use this for better line detection threshold. uses page median now.
  const makeHistogram = histogram();
  let _fontHeights = flatten<TextItemToDisplay>(pages.map(p => p.text)).map(t =>
    roundedToFixed(t.fontHeight, 2)
  );
  let fontHeights = unique(_fontHeights).sort();
  let height2color = fontHeights.reduce((res, height, ix) => {
    return { ...res, [height + ""]: brewer12[ix % 12] };
  }, {});

  let _fontNames = flatten<TextItemToDisplay>(pages.map(p => p.text)).map(
    t => t.style.fontFamily
  );

  let fontNames = unique(_fontNames).sort();
  let fontNames2color = fontNames.reduce((res, name, ix) => {
    return { ...res, [name + ""]: brewer12[ix % 12] };
  }, {});
};

export const getLeftEdgeOfColumns = (pages: PageToDisplay[]) => {
  const leftXs = flatten<TextItemToDisplay>(pages.map(p => p.text)).map(
    t => t.left
  );
  const makeHistogram = histogram();
  const leftXHist = makeHistogram(leftXs);
  const leftXBinCounts = leftXHist.map(x => x.length);
  if (leftXBinCounts) {
    const leftXMean = mean(leftXBinCounts) || NaN;
    const leftXStd = deviation(leftXBinCounts) || NaN;
    const leftXZscore = leftXBinCounts.map(x => (x - leftXMean) / leftXStd);
    const zThresh = 1;
    const columnLefts = leftXBinCounts.reduce((all, _val, ix) => {
      if (leftXZscore[ix] > zThresh) {
        all.push(median(leftXHist[ix]));
        return all;
      } else {
        return all;
      }
    }, []);
    return columnLefts;
  }
  return [NaN];
};

export const loadPageJson = async (
  dir: string,
  filePrefix: "textToDisplay" | "linesOfText",
  pageNumbers: number[] = []
) => {
  await existsElseMake(
    path.join(dir, filePrefix + ".json"),
    preprocessPdfs([dir])
  );

  const jsons = await ls(`${dir}/${filePrefix}-page*.json`);
  let pages = [];
  for (let j of jsons.sort()) {
    const page: PageToDisplay = await jsonfile.readFile(j);
    pages.push(page);
  }
  return pages; // sorted by page number
};

export const getLines = (
  columnLefts: number[],
  textItems: TextItemToDisplay[],
  pageNumber: number
) => {
  // so we've got the left side of columns detected by now
  // in a column we get all y values of text items
  // then sort the y vals, and combine y vals within some dist of eachother
  // then sort by x coord to get text order for items in a line

  const nCols = columnLefts.length;
  const textByColumn = columnLefts.map((left, i) => {
    const rightEdge = i < nCols - 1 ? columnLefts[i + 1] : Infinity;
    return textItems.filter(t => {
      const textLeft = Math.round(t.left);
      return left <= textLeft && textLeft < rightEdge && t.str !== " ";
    }); // removing spaces here, may need these for later formating
  });

  const medianFontHeight = Math.round(
    // @ts-ignore
    median(
      textItems.map(t => {
        return t.fontHeight;
      })
    )
  );

  let columnsLinesTextItems = [];
  for (var col of textByColumn) {
    const uniqueTops = [...new Set(col.map(t => Math.round(t.top)))].sort();
    let firstLine = col.find(x => Math.round(x.top) === uniqueTops[0]);
    let loopState = { count: 0, lines: [[firstLine]] };

    // combine tops within threshold
    const threshold = medianFontHeight / 2;
    for (let i = 1; i < uniqueTops.length; i++) {
      const prev = uniqueTops[i - 1];
      const current = uniqueTops[i];
      const textItems = col.filter(x => Math.round(x.top) === current);
      if (Math.abs(prev - current) < threshold) {
        loopState.lines[loopState.count].push(...textItems);
      } else {
        loopState.lines[loopState.count].sort(sortBy("left"));
        // if need performance, combine textitems here
        loopState.count++;
        loopState.lines.push([]);
        loopState.lines[loopState.count].push(...textItems);
      }
    }
    columnsLinesTextItems.push(loopState.lines);
  }

  // combine text items into a line with bounding box
  const linesInColumns: LineOfText[][] = columnsLinesTextItems.map(
    (col, colIx) => {
      return col.map((line, lineIndex) => {
        const nTextItems = line.length;
        return line.reduce(
          (all, text, i) => {
            if (!text) return all;
            if (i === 0) {
              // first
              return {
                id: `line${lineIndex}-col${colIx}`,
                pageNumber: pageNumber,
                columnIndex: colIx,
                lineIndex: lineIndex,
                left: text.left,
                top: text.top,
                height: text.transform[0],
                width: text.width,
                text: text.str,
                textIds: [text.id]
              };
            } else if (i === nTextItems - 1 && nTextItems > 1) {
              return {
                ...all,
                width: text.left + text.width - all.left,
                text: all.text + text.str,
                textIds: all.textIds.concat(text.id)
              };
            } else {
              // middle
              return {
                ...all,
                top: Math.min(text.top, all.top),
                height: Math.max(text.transform[0], all.height),
                text: all.text + text.str,
                textIds: all.textIds.concat(text.id)
              };
            }
          },
          {} as LineOfText
        );
      });
    }
  );

  linesInColumns.forEach(col => {
    col.sort(sortBy("top"));
  });

  return flatten<LineOfText>(linesInColumns);
};

export const getImageFiles = async (page, viewport) => {
  const opList = await page.getOperatorList();
  let svgGfx = new pdfjs.SVGGraphics(page.commonObjs, page.objs);
  svgGfx.embedFonts = true;
  const svg = await svgGfx.getSVG(opList, viewport); //in svg:img elements

  const imgs = svg.querySelectorAll("svg image") as HTMLOrSVGImageElement[];
  // document.body.append(svg)
  let images = [] as Image[];
  for (let img of imgs) {
    if (!img) continue;
    images.push({
      x: img.getAttribute("x") * scale,
      y: img.getAttribute("y") * scale,
      width: img.getAttribute("width") * scale,
      height: img.getAttribute("height") * scale,
      "xlink:href": img.getAttribute("xlink:href"),
      transform: img.getAttribute("transform"),
      gTransform: img.parentNode.getAttribute("transform")
    });
  }
};

export const loadPdfPages = async (
  path: string,
  pageNumbersToLoad: number[] = [],
  scale = 1
) => {
  const pdf = await pdfjs.getDocument(path);
  const allPageNumbers = [...Array(pdf.numPages).keys()].map(x => x + 1);
  const willLoadAllPages = pageNumbersToLoad.length === 0;
  const pageNumPropsOk =
    !willLoadAllPages &&
    Math.min(...pageNumbersToLoad) >= 0 &&
    Math.max(...pageNumbersToLoad) <= Math.max(...allPageNumbers);

  let pageNumbers;
  if (willLoadAllPages) {
    pageNumbers = allPageNumbers;
  } else {
    pageNumbers = pageNumPropsOk ? pageNumbersToLoad : allPageNumbers;
  }

  let pages = [] as _pdfjs.PDFPageProxy[];
  for (const pageNumber of pageNumbers) {
    const page = await pdf.getPage(pageNumber);
    pages.push(page);
  }
  return pages;
};

export interface Image {
  x: string;
  y: string;
  width: string;
  height: string;
  "xlink:href": string;
  transform: string;
  gTransform: string;
}

export type PageToDisplay = {
  pageNumber: number;
  text: TextItemToDisplay[];
  viewportFlat: {
    width: number;
    height: number;
    xMin: number;
    yMin: number;
    xMax: number;
    yMax: number;
  };
};

export type TextItemToDisplay = {
  str: string; // the text
  dir: string; // text direction
  width: number;
  transform: number[]; // [fontheight, rotation?, rotation?, fontwidth, x, y ]
  id: string; // we made this
  top: number;
  left: number;
  fallbackFontName: string;
  style: { fontFamily: string; ascent: number; descent: number };
  fontHeight: number;
  fontWidth: number;
  scaleX: number;
};

export interface LineOfText {
  id: string;
  // pageId: number;
  lineIndex: number;
  columnIndex: number;
  left: number;
  top: number;
  width: number;
  height: number;
  text: string;
  textIds: string[];
}
