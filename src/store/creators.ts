import { mergeDefaults } from "../renderer/utils";
import { number } from "prop-types";
import uuidv1 = require("uuid/v1");

// todo! nodes are now frames
// viewbox left/top should be scrollLeft/scrollTop

export type NodeDataTypes =
  | "empty"
  | "userHtml" // html made by user after writing
  | "pdf.segment.viewbox" //
  | "pdf.segment.text"
  | "pdf.publication" //
  | "user"
  | "person"
  | "venue"
  | "query" // queries have style overrides, combine subqueries to reuse, ooo
  | "projection/map/affinity/dimension/coordinates matter";

export interface NodeMeta {
  createdBy: string;
  timeCreated: number;
  timeUpdated: number;
  editors?: string[];
}

export interface NodeBase {
  id: string;
  data: { type: NodeDataTypes };
  style: { left: number; top: number; width: number; height: number };
  meta: NodeMeta;
}

export interface Empty extends NodeBase {
  data: {
    type: "empty";
  };
  meta: NodeMeta;
}

export interface PdfPathInfo {
  pdfPath: string;
  pdfName: string;
  dir: string;
}

const ViewboxDataDefault = {
  left: 0,
  top: 0,
  height: 0,
  width: 0,
  userId: "default",
  pdfDir: "",
  pageNumber: 0,
  type: "pdf.segment.viewbox" as NodeDataTypes,
  scale: 1
};
export type ViewboxData = typeof ViewboxDataDefault;
export interface PdfSegmentViewbox extends NodeBase {
  data: ViewboxData;
}
import { CircleConfig, LineConfig } from "konva";
export const makePdfSegmentViewbox = (
  viewbox = {} as Partial<ViewboxData>,
  style = {}
) => {
  const now = Date.now();
  const id = uuidv1();
  console.log({ ...ViewboxDataDefault, ...viewbox });

  return {
    id: id,
    data: { ...ViewboxDataDefault, ...viewbox },
    style: {
      id: id,
      type: "circle",
      left: Math.random() * 200 + 20,
      top: Math.random() * 200 + 20,
      width: 200,
      height: 200,
      fill: "blue",
      draggabled: true,
      radius: 5,
      stroke: "blue",
      strokeWidth: 4,
      ...style
    },
    meta: makeNodeMeta()
  } as PdfSegmentViewbox;
};

const makeNodeMeta = (meta = {}): NodeMeta => {
  // pass in existing meta to updatetime
  const now = Date.now();
  return {
    createdBy: "defaultUser",
    timeCreated: now,
    ...meta,
    timeUpdated: now
  };
};

const PdfPublicationDefaults = {
  id: "",
  data: {
    type: "pdf.publication" as NodeDataTypes,
    publicationType: "", // Journal Article, Conference proceedings, book
    pdfDir: "sameAsId",
    fileExt: ".pdf", // saveAsId.pdf
    title: "",
    venue: "",
    authors: [] as string[],
    year: NaN as number,
    volume: "" as string | number,
    issue: "" as string | number,
    url: "",
    arxivId: "",
    pmid: "",
    doi: "",
    isbn: "",
    issn: "",
    published: true
  },
  style: {
    id: "",
    left: Math.random() * 200 + 20,
    top: Math.random() * 200 + 20,
    width: 200,
    height: 200,
    fill: "grey",
    stroke: "red"
  },
  meta: makeNodeMeta()
};
export type PdfPublication = typeof PdfPublicationDefaults;

export const makePdfPublication = (dirName: string, data = {}, style = {}) => {
  return {
    ...PdfPublicationDefaults,
    id: dirName,
    data: { ...PdfPublicationDefaults.data, ...data },
    style: { ...PdfPublicationDefaults.style, ...style, id: dirName }
  };
};

const LinkDefaults = {
  id: "",
  data: { text: "", html: "" },
  style: {
    stroke: "lightgrey"
  },
  meta: makeNodeMeta(),
  source: "",
  target: "",
  isDirected: true
};

export type LinkBase = typeof LinkDefaults;

export const makeLink = (sourceId: string, targetId: string, data = {}) => {
  const id = "link-" + uuidv1();
  return {
    ...LinkDefaults,
    id,
    source: sourceId,
    target: targetId,
    style: { ...LinkDefaults.style },
    data: { ...LinkDefaults.data, ...data }
  };
};

const UserHtmlDefaults = {
  id: "",
  data: { type: "userHtml" as NodeDataTypes, html: "", text: "" },
  meta: makeNodeMeta(),
  style: {
    left: 0,
    top: 0,
    width: 200,
    height: 220
  }
};
export type UserHtml = typeof UserHtmlDefaults;
export const makeUserHtml = (props = { data: {}, style: {} }) => {
  const data = { html: "", text: "", ...props.data };
  const id = uuidv1();
  return {
    ...UserHtmlDefaults,
    id,
    data: { ...UserHtmlDefaults.data, ...data },
    style: { ...UserHtmlDefaults.style, ...props.style }
  };
};

export type aNode = PdfSegmentViewbox | Empty | UserHtml;
export type aLink = LinkBase;
export type Nodes = { [id: string]: aNode }; // or...
export type Links = { [id: string]: aLink }; // or...
