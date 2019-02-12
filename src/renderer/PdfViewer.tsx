import * as React from "react";

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
import jsonfile = require("jsonfile");
import path = require("path");

import PageCanvas from "./PageCanvas";
import PageText from "./PageText";
import PageSvg from "./PageSvg";

import {
  flatten,
  midPoint,
  getRectCoords,
  sortBy,
  unique,
  brewer12
} from "./utils";
import { histogram, mean, median, deviation } from "d3-array";
import produce from "immer";
import { loadPdfPages, loadPageJson, PageOfText } from "./io";

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

export interface Image {
  x: string;
  y: string;
  width: string;
  height: string;
  "xlink:href": string;
  transform: string;
  gTransform: string;
}

interface Page {
  pageNumber: number;
  viewport: any; // pdfjs.PDFPageViewport;
  text: PageOfText;
  page: any; // pdfjs.PDFPageProxy;
  linesOfText: LineOfText[];
  // images: Image[];
}


import styled from "styled-components";
import { PdfPathInfo } from "../store/createStore";

/**
 * @class **PdfViewer**
 * todo zoom, file name prop, layer props, keyboard shortcuts
 */
const PdfViewerDefaults = {
  props: {
    pageNumbersToLoad: [] as number[],
    pathInfo: {} as PdfPathInfo,
    viewBox: {
      top: 110,
      left: 110,
      width: "100%" as number | string | undefined,
      height: "100%" as number | string | undefined
    }
  },
  state: {
    scale: 2, // todo scale
    pages: [] as Page[],
    columnLefts: [] as number[],
    height2color: {} as any,
    fontNames2color: {} as any,
    meta: {} as {
      info: PDFInfo;
      metadata: PDFMetadata;
    },
    outline: [] as PDFTreeNode[]
  }
};

export default class PdfViewer extends React.Component<
  typeof PdfViewerDefaults.props,
  typeof PdfViewerDefaults.state
> {
  static defaultProps = PdfViewerDefaults.props;
  state = PdfViewerDefaults.state;
  scrollRef = React.createRef<HTMLDivElement>()

  async componentDidMount() {
    await this.loadFiles();
    const {left, top} = this.props.viewBox
    this.scrollRef.current.scrollTo(left, top)
  }

  loadFiles = async () => {
    this.setState({ pages: [] });
    const {
      pathInfo: { pdfName, pdfPath, dir },
      pageNumbersToLoad
    } = this.props;

    const [
      pdfPages,
      linesOfText,
      textToDisplay,
      columnLefts
    ] = await Promise.all([
      loadPdfPages(pdfPath, pageNumbersToLoad),
      loadPageJson(dir, "linesOfText", pageNumbersToLoad),
      loadPageJson(dir, "textToDisplay", pageNumbersToLoad),
      jsonfile.readFile(path.join(dir, "columnLefts.json"))
    ]);

    let pages = [] as Page[];
    for (let i in pdfPages) {
      pages.push({
        linesOfText: linesOfText[i],
        page: pdfPages[i],
        pageNumber: pageNumbersToLoad[i],
        text: textToDisplay[i],
        viewport: pdfPages[i].getViewport(this.state.scale)
      });
    }
    if (this.state.scale !== 1) {
      const scaledPages = this.scalePages(pages, 1, this.state.scale);
      this.setState({ pages: scaledPages, columnLefts });
    } else {
      this.setState({ pages, columnLefts });
    }
  };

  scale = (obj, keyNames: string[], prevScale, scale) => {
    const scaled = keyNames.reduce((all, keyName, ix) => {
      if (!obj.hasOwnProperty(keyName)) return all;
      return { ...all, [keyName]: (obj[keyName] / prevScale) * scale };
    }, {});

    return scaled;
  };

  scalePages = (pages: Page[], prevScale: number = 1, scale: number = 1) => {
    let keysToScale = ["height", "left", "top", "width"];

    let scaledPages = [] as Page[];
    for (let [ix, page] of pages.entries()) {
      const linesOfText = page.linesOfText.map(lot => {
        return {
          ...lot,
          ...this.scale(lot, keysToScale, prevScale, scale)
        };
      });
      const text = page.text.text.map(t => {
        return {
          ...t,
          ...this.scale(t, keysToScale, prevScale, scale)
        };
      });
      const viewport = page.page.getViewport(scale);
      scaledPages.push({ ...page, linesOfText, viewport });
    }
    return scaledPages;
  };

  async componentDidUpdate(prevProps: typeof PdfViewerDefaults.props) {
    if (prevProps.pathInfo !== this.props.pathInfo) {
      await this.loadFiles();
    }
  }

  zoom = (e: React.WheelEvent<HTMLDivElement>) => {
    if (e.ctrlKey) {
      const deltaY = e.deltaY;
      this.setState(state => {
        const prevScale = this.state.scale;
        const newScale = prevScale - deltaY / 1000;
        const scaledPages = this.scalePages(state.pages, prevScale, newScale);
        return { pages: scaledPages, scale: newScale };
      });
    }
  };

  renderPages = () => {
    const { pages } = this.state;
    const havePages = pages.length > 0;
    if (!havePages) return null;
    return pages.map((page, pageNum) => {
      const { width, height } = page.viewport;
      return (
        <div
          key={pageNum}
          onWheel={this.zoom}
          style={{ width, height, position: "relative" }}
        >
          <PageCanvas
            key={"canvas-" + pageNum}
            page={page.page}
            viewport={page.viewport}
          />
          {/* <PageText
                  key={"text-" + pageNum}
                  scale={this.state.scale}
                  pageOfText={page.text}
                  // height={height}
                /> */}
          <PageSvg
            // scale={this.state.scale}
            key={"svg-" + pageNum}
            svgWidth={width}
            svgHeight={height}
            pageOfText={page.text}
            columnLefts={this.state.columnLefts.map(x => x * this.state.scale)}
            linesOfText={page.linesOfText}
            // images={page.images}
            height2color={this.state.height2color}
            fontNames2color={this.state.fontNames2color}
            pdfPathInfo={this.props.pathInfo}
            pageNumber={pageNum}
          />
        </div>
      );
    });
  };

  render() {
    const {width, height} = this.props.viewBox

    // todo: set height and width and then scrollto
    return (
      <div
        ref={this.scrollRef}
        style={{
          maxWidth: width,
          maxHeight: height,
          boxSizing: "border-box",
          overflow: "scroll"
        }}
      >
        {this.renderPages()}
      </div>
    );
  }
}