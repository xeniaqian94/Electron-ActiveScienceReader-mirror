/**
 * @licstart The following is the entire license notice for the
 * Javascript code in this page
 *
 * Copyright 2019 Mozilla Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @licend The above is the entire license notice for the
 * Javascript code in this page
 */
"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.PDFNetworkStream = PDFNetworkStream;
exports.NetworkManager = NetworkManager;

var _regenerator = _interopRequireDefault(require("@babel/runtime/regenerator"));

var _util = require("../shared/util");

var _network_utils = require("./network_utils");

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function asyncGeneratorStep(gen, resolve, reject, _next, _throw, key, arg) { try { var info = gen[key](arg); var value = info.value; } catch (error) { reject(error); return; } if (info.done) { resolve(value); } else { Promise.resolve(value).then(_next, _throw); } }

function _asyncToGenerator(fn) { return function () { var self = this, args = arguments; return new Promise(function (resolve, reject) { var gen = fn.apply(self, args); function _next(value) { asyncGeneratorStep(gen, resolve, reject, _next, _throw, "next", value); } function _throw(err) { asyncGeneratorStep(gen, resolve, reject, _next, _throw, "throw", err); } _next(undefined); }); }; }

;
var OK_RESPONSE = 200;
var PARTIAL_CONTENT_RESPONSE = 206;

function NetworkManager(url, args) {
  this.url = url;
  args = args || {};
  this.isHttp = /^https?:/i.test(url);
  this.httpHeaders = this.isHttp && args.httpHeaders || {};
  this.withCredentials = args.withCredentials || false;

  this.getXhr = args.getXhr || function NetworkManager_getXhr() {
    return new XMLHttpRequest();
  };

  this.currXhrId = 0;
  this.pendingRequests = Object.create(null);
}

function getArrayBuffer(xhr) {
  var data = xhr.response;

  if (typeof data !== 'string') {
    return data;
  }

  var array = (0, _util.stringToBytes)(data);
  return array.buffer;
}

NetworkManager.prototype = {
  requestRange: function NetworkManager_requestRange(begin, end, listeners) {
    var args = {
      begin: begin,
      end: end
    };

    for (var prop in listeners) {
      args[prop] = listeners[prop];
    }

    return this.request(args);
  },
  requestFull: function NetworkManager_requestFull(listeners) {
    return this.request(listeners);
  },
  request: function NetworkManager_request(args) {
    var xhr = this.getXhr();
    var xhrId = this.currXhrId++;
    var pendingRequest = this.pendingRequests[xhrId] = {
      xhr: xhr
    };
    xhr.open('GET', this.url);
    xhr.withCredentials = this.withCredentials;

    for (var property in this.httpHeaders) {
      var value = this.httpHeaders[property];

      if (typeof value === 'undefined') {
        continue;
      }

      xhr.setRequestHeader(property, value);
    }

    if (this.isHttp && 'begin' in args && 'end' in args) {
      var rangeStr = args.begin + '-' + (args.end - 1);
      xhr.setRequestHeader('Range', 'bytes=' + rangeStr);
      pendingRequest.expectedStatus = 206;
    } else {
      pendingRequest.expectedStatus = 200;
    }

    xhr.responseType = 'arraybuffer';

    if (args.onError) {
      xhr.onerror = function (evt) {
        args.onError(xhr.status);
      };
    }

    xhr.onreadystatechange = this.onStateChange.bind(this, xhrId);
    xhr.onprogress = this.onProgress.bind(this, xhrId);
    pendingRequest.onHeadersReceived = args.onHeadersReceived;
    pendingRequest.onDone = args.onDone;
    pendingRequest.onError = args.onError;
    pendingRequest.onProgress = args.onProgress;
    xhr.send(null);
    return xhrId;
  },
  onProgress: function NetworkManager_onProgress(xhrId, evt) {
    var pendingRequest = this.pendingRequests[xhrId];

    if (!pendingRequest) {
      return;
    }

    if (pendingRequest.onProgress) {
      pendingRequest.onProgress(evt);
    }
  },
  onStateChange: function NetworkManager_onStateChange(xhrId, evt) {
    var pendingRequest = this.pendingRequests[xhrId];

    if (!pendingRequest) {
      return;
    }

    var xhr = pendingRequest.xhr;

    if (xhr.readyState >= 2 && pendingRequest.onHeadersReceived) {
      pendingRequest.onHeadersReceived();
      delete pendingRequest.onHeadersReceived;
    }

    if (xhr.readyState !== 4) {
      return;
    }

    if (!(xhrId in this.pendingRequests)) {
      return;
    }

    delete this.pendingRequests[xhrId];

    if (xhr.status === 0 && this.isHttp) {
      if (pendingRequest.onError) {
        pendingRequest.onError(xhr.status);
      }

      return;
    }

    var xhrStatus = xhr.status || OK_RESPONSE;
    var ok_response_on_range_request = xhrStatus === OK_RESPONSE && pendingRequest.expectedStatus === PARTIAL_CONTENT_RESPONSE;

    if (!ok_response_on_range_request && xhrStatus !== pendingRequest.expectedStatus) {
      if (pendingRequest.onError) {
        pendingRequest.onError(xhr.status);
      }

      return;
    }

    var chunk = getArrayBuffer(xhr);

    if (xhrStatus === PARTIAL_CONTENT_RESPONSE) {
      var rangeHeader = xhr.getResponseHeader('Content-Range');
      var matches = /bytes (\d+)-(\d+)\/(\d+)/.exec(rangeHeader);
      var begin = parseInt(matches[1], 10);
      pendingRequest.onDone({
        begin: begin,
        chunk: chunk
      });
    } else if (chunk) {
      pendingRequest.onDone({
        begin: 0,
        chunk: chunk
      });
    } else if (pendingRequest.onError) {
      pendingRequest.onError(xhr.status);
    }
  },
  hasPendingRequests: function NetworkManager_hasPendingRequests() {
    for (var xhrId in this.pendingRequests) {
      return true;
    }

    return false;
  },
  getRequestXhr: function NetworkManager_getXhr(xhrId) {
    return this.pendingRequests[xhrId].xhr;
  },
  isPendingRequest: function NetworkManager_isPendingRequest(xhrId) {
    return xhrId in this.pendingRequests;
  },
  abortAllRequests: function NetworkManager_abortAllRequests() {
    for (var xhrId in this.pendingRequests) {
      this.abortRequest(xhrId | 0);
    }
  },
  abortRequest: function NetworkManager_abortRequest(xhrId) {
    var xhr = this.pendingRequests[xhrId].xhr;
    delete this.pendingRequests[xhrId];
    xhr.abort();
  }
};

function PDFNetworkStream(source) {
  this._source = source;
  this._manager = new NetworkManager(source.url, {
    httpHeaders: source.httpHeaders,
    withCredentials: source.withCredentials
  });
  this._rangeChunkSize = source.rangeChunkSize;
  this._fullRequestReader = null;
  this._rangeRequestReaders = [];
}

PDFNetworkStream.prototype = {
  _onRangeRequestReaderClosed: function PDFNetworkStream_onRangeRequestReaderClosed(reader) {
    var i = this._rangeRequestReaders.indexOf(reader);

    if (i >= 0) {
      this._rangeRequestReaders.splice(i, 1);
    }
  },
  getFullReader: function PDFNetworkStream_getFullReader() {
    (0, _util.assert)(!this._fullRequestReader);
    this._fullRequestReader = new PDFNetworkStreamFullRequestReader(this._manager, this._source);
    return this._fullRequestReader;
  },
  getRangeReader: function PDFNetworkStream_getRangeReader(begin, end) {
    var reader = new PDFNetworkStreamRangeRequestReader(this._manager, begin, end);
    reader.onClosed = this._onRangeRequestReaderClosed.bind(this);

    this._rangeRequestReaders.push(reader);

    return reader;
  },
  cancelAllRequests: function PDFNetworkStream_cancelAllRequests(reason) {
    if (this._fullRequestReader) {
      this._fullRequestReader.cancel(reason);
    }

    var readers = this._rangeRequestReaders.slice(0);

    readers.forEach(function (reader) {
      reader.cancel(reason);
    });
  }
};

function PDFNetworkStreamFullRequestReader(manager, source) {
  this._manager = manager;
  var args = {
    onHeadersReceived: this._onHeadersReceived.bind(this),
    onDone: this._onDone.bind(this),
    onError: this._onError.bind(this),
    onProgress: this._onProgress.bind(this)
  };
  this._url = source.url;
  this._fullRequestId = manager.requestFull(args);
  this._headersReceivedCapability = (0, _util.createPromiseCapability)();
  this._disableRange = source.disableRange || false;
  this._contentLength = source.length;
  this._rangeChunkSize = source.rangeChunkSize;

  if (!this._rangeChunkSize && !this._disableRange) {
    this._disableRange = true;
  }

  this._isStreamingSupported = false;
  this._isRangeSupported = false;
  this._cachedChunks = [];
  this._requests = [];
  this._done = false;
  this._storedError = undefined;
  this._filename = null;
  this.onProgress = null;
}

PDFNetworkStreamFullRequestReader.prototype = {
  _onHeadersReceived: function PDFNetworkStreamFullRequestReader_onHeadersReceived() {
    var fullRequestXhrId = this._fullRequestId;

    var fullRequestXhr = this._manager.getRequestXhr(fullRequestXhrId);

    var getResponseHeader = function getResponseHeader(name) {
      return fullRequestXhr.getResponseHeader(name);
    };

    var _validateRangeRequest = (0, _network_utils.validateRangeRequestCapabilities)({
      getResponseHeader: getResponseHeader,
      isHttp: this._manager.isHttp,
      rangeChunkSize: this._rangeChunkSize,
      disableRange: this._disableRange
    }),
        allowRangeRequests = _validateRangeRequest.allowRangeRequests,
        suggestedLength = _validateRangeRequest.suggestedLength;

    if (allowRangeRequests) {
      this._isRangeSupported = true;
    }

    this._contentLength = suggestedLength || this._contentLength;
    this._filename = (0, _network_utils.extractFilenameFromHeader)(getResponseHeader);

    if (this._isRangeSupported) {
      this._manager.abortRequest(fullRequestXhrId);
    }

    this._headersReceivedCapability.resolve();
  },
  _onDone: function PDFNetworkStreamFullRequestReader_onDone(args) {
    if (args) {
      if (this._requests.length > 0) {
        var requestCapability = this._requests.shift();

        requestCapability.resolve({
          value: args.chunk,
          done: false
        });
      } else {
        this._cachedChunks.push(args.chunk);
      }
    }

    this._done = true;

    if (this._cachedChunks.length > 0) {
      return;
    }

    this._requests.forEach(function (requestCapability) {
      requestCapability.resolve({
        value: undefined,
        done: true
      });
    });

    this._requests = [];
  },
  _onError: function PDFNetworkStreamFullRequestReader_onError(status) {
    var url = this._url;
    var exception = (0, _network_utils.createResponseStatusError)(status, url);
    this._storedError = exception;

    this._headersReceivedCapability.reject(exception);

    this._requests.forEach(function (requestCapability) {
      requestCapability.reject(exception);
    });

    this._requests = [];
    this._cachedChunks = [];
  },
  _onProgress: function PDFNetworkStreamFullRequestReader_onProgress(data) {
    if (this.onProgress) {
      this.onProgress({
        loaded: data.loaded,
        total: data.lengthComputable ? data.total : this._contentLength
      });
    }
  },

  get filename() {
    return this._filename;
  },

  get isRangeSupported() {
    return this._isRangeSupported;
  },

  get isStreamingSupported() {
    return this._isStreamingSupported;
  },

  get contentLength() {
    return this._contentLength;
  },

  get headersReady() {
    return this._headersReceivedCapability.promise;
  },

  read: function () {
    var _read = _asyncToGenerator(
    /*#__PURE__*/
    _regenerator.default.mark(function _callee() {
      var chunk, requestCapability;
      return _regenerator.default.wrap(function _callee$(_context) {
        while (1) {
          switch (_context.prev = _context.next) {
            case 0:
              if (!this._storedError) {
                _context.next = 2;
                break;
              }

              throw this._storedError;

            case 2:
              if (!(this._cachedChunks.length > 0)) {
                _context.next = 5;
                break;
              }

              chunk = this._cachedChunks.shift();
              return _context.abrupt("return", {
                value: chunk,
                done: false
              });

            case 5:
              if (!this._done) {
                _context.next = 7;
                break;
              }

              return _context.abrupt("return", {
                value: undefined,
                done: true
              });

            case 7:
              requestCapability = (0, _util.createPromiseCapability)();

              this._requests.push(requestCapability);

              return _context.abrupt("return", requestCapability.promise);

            case 10:
            case "end":
              return _context.stop();
          }
        }
      }, _callee, this);
    }));

    function read() {
      return _read.apply(this, arguments);
    }

    return read;
  }(),
  cancel: function PDFNetworkStreamFullRequestReader_cancel(reason) {
    this._done = true;

    this._headersReceivedCapability.reject(reason);

    this._requests.forEach(function (requestCapability) {
      requestCapability.resolve({
        value: undefined,
        done: true
      });
    });

    this._requests = [];

    if (this._manager.isPendingRequest(this._fullRequestId)) {
      this._manager.abortRequest(this._fullRequestId);
    }

    this._fullRequestReader = null;
  }
};

function PDFNetworkStreamRangeRequestReader(manager, begin, end) {
  this._manager = manager;
  var args = {
    onDone: this._onDone.bind(this),
    onProgress: this._onProgress.bind(this)
  };
  this._requestId = manager.requestRange(begin, end, args);
  this._requests = [];
  this._queuedChunk = null;
  this._done = false;
  this.onProgress = null;
  this.onClosed = null;
}

PDFNetworkStreamRangeRequestReader.prototype = {
  _close: function PDFNetworkStreamRangeRequestReader_close() {
    if (this.onClosed) {
      this.onClosed(this);
    }
  },
  _onDone: function PDFNetworkStreamRangeRequestReader_onDone(data) {
    var chunk = data.chunk;

    if (this._requests.length > 0) {
      var requestCapability = this._requests.shift();

      requestCapability.resolve({
        value: chunk,
        done: false
      });
    } else {
      this._queuedChunk = chunk;
    }

    this._done = true;

    this._requests.forEach(function (requestCapability) {
      requestCapability.resolve({
        value: undefined,
        done: true
      });
    });

    this._requests = [];

    this._close();
  },
  _onProgress: function PDFNetworkStreamRangeRequestReader_onProgress(evt) {
    if (!this.isStreamingSupported && this.onProgress) {
      this.onProgress({
        loaded: evt.loaded
      });
    }
  },

  get isStreamingSupported() {
    return false;
  },

  read: function () {
    var _read2 = _asyncToGenerator(
    /*#__PURE__*/
    _regenerator.default.mark(function _callee2() {
      var chunk, requestCapability;
      return _regenerator.default.wrap(function _callee2$(_context2) {
        while (1) {
          switch (_context2.prev = _context2.next) {
            case 0:
              if (!(this._queuedChunk !== null)) {
                _context2.next = 4;
                break;
              }

              chunk = this._queuedChunk;
              this._queuedChunk = null;
              return _context2.abrupt("return", {
                value: chunk,
                done: false
              });

            case 4:
              if (!this._done) {
                _context2.next = 6;
                break;
              }

              return _context2.abrupt("return", {
                value: undefined,
                done: true
              });

            case 6:
              requestCapability = (0, _util.createPromiseCapability)();

              this._requests.push(requestCapability);

              return _context2.abrupt("return", requestCapability.promise);

            case 9:
            case "end":
              return _context2.stop();
          }
        }
      }, _callee2, this);
    }));

    function read() {
      return _read2.apply(this, arguments);
    }

    return read;
  }(),
  cancel: function PDFNetworkStreamRangeRequestReader_cancel(reason) {
    this._done = true;

    this._requests.forEach(function (requestCapability) {
      requestCapability.resolve({
        value: undefined,
        done: true
      });
    });

    this._requests = [];

    if (this._manager.isPendingRequest(this._requestId)) {
      this._manager.abortRequest(this._requestId);
    }

    this._close();
  }
};