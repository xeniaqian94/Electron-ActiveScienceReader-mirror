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
'use strict';

function _typeof(obj) { if (typeof Symbol === "function" && typeof Symbol.iterator === "symbol") { _typeof = function _typeof(obj) { return typeof obj; }; } else { _typeof = function _typeof(obj) { return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj; }; } return _typeof(obj); }

var FontInspector = function FontInspectorClosure() {
  var fonts, createObjectURL;
  var active = false;
  var fontAttribute = 'data-font-name';

  function removeSelection() {
    var divs = document.querySelectorAll("span[".concat(fontAttribute, "]"));
    var _iteratorNormalCompletion = true;
    var _didIteratorError = false;
    var _iteratorError = undefined;

    try {
      for (var _iterator = divs[Symbol.iterator](), _step; !(_iteratorNormalCompletion = (_step = _iterator.next()).done); _iteratorNormalCompletion = true) {
        var div = _step.value;
        div.className = '';
      }
    } catch (err) {
      _didIteratorError = true;
      _iteratorError = err;
    } finally {
      try {
        if (!_iteratorNormalCompletion && _iterator.return != null) {
          _iterator.return();
        }
      } finally {
        if (_didIteratorError) {
          throw _iteratorError;
        }
      }
    }
  }

  function resetSelection() {
    var divs = document.querySelectorAll("span[".concat(fontAttribute, "]"));
    var _iteratorNormalCompletion2 = true;
    var _didIteratorError2 = false;
    var _iteratorError2 = undefined;

    try {
      for (var _iterator2 = divs[Symbol.iterator](), _step2; !(_iteratorNormalCompletion2 = (_step2 = _iterator2.next()).done); _iteratorNormalCompletion2 = true) {
        var div = _step2.value;
        div.className = 'debuggerHideText';
      }
    } catch (err) {
      _didIteratorError2 = true;
      _iteratorError2 = err;
    } finally {
      try {
        if (!_iteratorNormalCompletion2 && _iterator2.return != null) {
          _iterator2.return();
        }
      } finally {
        if (_didIteratorError2) {
          throw _iteratorError2;
        }
      }
    }
  }

  function selectFont(fontName, show) {
    var divs = document.querySelectorAll("span[".concat(fontAttribute, "=").concat(fontName, "]"));
    var _iteratorNormalCompletion3 = true;
    var _didIteratorError3 = false;
    var _iteratorError3 = undefined;

    try {
      for (var _iterator3 = divs[Symbol.iterator](), _step3; !(_iteratorNormalCompletion3 = (_step3 = _iterator3.next()).done); _iteratorNormalCompletion3 = true) {
        var div = _step3.value;
        div.className = show ? 'debuggerShowText' : 'debuggerHideText';
      }
    } catch (err) {
      _didIteratorError3 = true;
      _iteratorError3 = err;
    } finally {
      try {
        if (!_iteratorNormalCompletion3 && _iterator3.return != null) {
          _iterator3.return();
        }
      } finally {
        if (_didIteratorError3) {
          throw _iteratorError3;
        }
      }
    }
  }

  function textLayerClick(e) {
    if (!e.target.dataset.fontName || e.target.tagName.toUpperCase() !== 'SPAN') {
      return;
    }

    var fontName = e.target.dataset.fontName;
    var selects = document.getElementsByTagName('input');

    for (var i = 0; i < selects.length; ++i) {
      var select = selects[i];

      if (select.dataset.fontName !== fontName) {
        continue;
      }

      select.checked = !select.checked;
      selectFont(fontName, select.checked);
      select.scrollIntoView();
    }
  }

  return {
    id: 'FontInspector',
    name: 'Font Inspector',
    panel: null,
    manager: null,
    init: function init(pdfjsLib) {
      var panel = this.panel;
      panel.setAttribute('style', 'padding: 5px;');
      var tmp = document.createElement('button');
      tmp.addEventListener('click', resetSelection);
      tmp.textContent = 'Refresh';
      panel.appendChild(tmp);
      fonts = document.createElement('div');
      panel.appendChild(fonts);
      createObjectURL = pdfjsLib.createObjectURL;
    },
    cleanup: function cleanup() {
      fonts.textContent = '';
    },
    enabled: false,

    get active() {
      return active;
    },

    set active(value) {
      active = value;

      if (active) {
        document.body.addEventListener('click', textLayerClick, true);
        resetSelection();
      } else {
        document.body.removeEventListener('click', textLayerClick, true);
        removeSelection();
      }
    },

    fontAdded: function fontAdded(fontObj, url) {
      var _this = this;

      function properties(obj, list) {
        var moreInfo = document.createElement('table');

        for (var i = 0; i < list.length; i++) {
          var tr = document.createElement('tr');
          var td1 = document.createElement('td');
          td1.textContent = list[i];
          tr.appendChild(td1);
          var td2 = document.createElement('td');
          td2.textContent = obj[list[i]].toString();
          tr.appendChild(td2);
          moreInfo.appendChild(tr);
        }

        return moreInfo;
      }

      var moreInfo = properties(fontObj, ['name', 'type']);
      var fontName = fontObj.loadedName;
      var font = document.createElement('div');
      var name = document.createElement('span');
      name.textContent = fontName;
      var download = document.createElement('a');

      if (url) {
        url = /url\(['"]?([^\)"']+)/.exec(url);
        download.href = url[1];
      } else if (fontObj.data) {
        download.href = createObjectURL(fontObj.data, fontObj.mimeType);
      }

      download.textContent = 'Download';
      var logIt = document.createElement('a');
      logIt.href = '';
      logIt.textContent = 'Log';
      logIt.addEventListener('click', function (event) {
        event.preventDefault();
        console.log(fontObj);
      });
      var select = document.createElement('input');
      select.setAttribute('type', 'checkbox');
      select.dataset.fontName = fontName;
      select.addEventListener('click', function (select, fontName) {
        return function () {
          selectFont(fontName, select.checked);
        };
      }(select, fontName));
      font.appendChild(select);
      font.appendChild(name);
      font.appendChild(document.createTextNode(' '));
      font.appendChild(download);
      font.appendChild(document.createTextNode(' '));
      font.appendChild(logIt);
      font.appendChild(moreInfo);
      fonts.appendChild(font);
      setTimeout(function () {
        if (_this.active) {
          resetSelection();
        }
      }, 2000);
    }
  };
}();

var opMap;

var StepperManager = function StepperManagerClosure() {
  var steppers = [];
  var stepperDiv = null;
  var stepperControls = null;
  var stepperChooser = null;
  var breakPoints = Object.create(null);
  return {
    id: 'Stepper',
    name: 'Stepper',
    panel: null,
    manager: null,
    init: function init(pdfjsLib) {
      var self = this;
      this.panel.setAttribute('style', 'padding: 5px;');
      stepperControls = document.createElement('div');
      stepperChooser = document.createElement('select');
      stepperChooser.addEventListener('change', function (event) {
        self.selectStepper(this.value);
      });
      stepperControls.appendChild(stepperChooser);
      stepperDiv = document.createElement('div');
      this.panel.appendChild(stepperControls);
      this.panel.appendChild(stepperDiv);

      if (sessionStorage.getItem('pdfjsBreakPoints')) {
        breakPoints = JSON.parse(sessionStorage.getItem('pdfjsBreakPoints'));
      }

      opMap = Object.create(null);

      for (var key in pdfjsLib.OPS) {
        opMap[pdfjsLib.OPS[key]] = key;
      }
    },
    cleanup: function cleanup() {
      stepperChooser.textContent = '';
      stepperDiv.textContent = '';
      steppers = [];
    },
    enabled: false,
    active: false,
    create: function create(pageIndex) {
      var debug = document.createElement('div');
      debug.id = 'stepper' + pageIndex;
      debug.setAttribute('hidden', true);
      debug.className = 'stepper';
      stepperDiv.appendChild(debug);
      var b = document.createElement('option');
      b.textContent = 'Page ' + (pageIndex + 1);
      b.value = pageIndex;
      stepperChooser.appendChild(b);
      var initBreakPoints = breakPoints[pageIndex] || [];
      var stepper = new Stepper(debug, pageIndex, initBreakPoints);
      steppers.push(stepper);

      if (steppers.length === 1) {
        this.selectStepper(pageIndex, false);
      }

      return stepper;
    },
    selectStepper: function selectStepper(pageIndex, selectPanel) {
      var i;
      pageIndex = pageIndex | 0;

      if (selectPanel) {
        this.manager.selectPanel(this);
      }

      for (i = 0; i < steppers.length; ++i) {
        var stepper = steppers[i];

        if (stepper.pageIndex === pageIndex) {
          stepper.panel.removeAttribute('hidden');
        } else {
          stepper.panel.setAttribute('hidden', true);
        }
      }

      var options = stepperChooser.options;

      for (i = 0; i < options.length; ++i) {
        var option = options[i];
        option.selected = (option.value | 0) === pageIndex;
      }
    },
    saveBreakPoints: function saveBreakPoints(pageIndex, bps) {
      breakPoints[pageIndex] = bps;
      sessionStorage.setItem('pdfjsBreakPoints', JSON.stringify(breakPoints));
    }
  };
}();

var Stepper = function StepperClosure() {
  function c(tag, textContent) {
    var d = document.createElement(tag);

    if (textContent) {
      d.textContent = textContent;
    }

    return d;
  }

  function simplifyArgs(args) {
    if (typeof args === 'string') {
      var MAX_STRING_LENGTH = 75;
      return args.length <= MAX_STRING_LENGTH ? args : args.substring(0, MAX_STRING_LENGTH) + '...';
    }

    if (_typeof(args) !== 'object' || args === null) {
      return args;
    }

    if ('length' in args) {
      var simpleArgs = [],
          i,
          ii;
      var MAX_ITEMS = 10;

      for (i = 0, ii = Math.min(MAX_ITEMS, args.length); i < ii; i++) {
        simpleArgs.push(simplifyArgs(args[i]));
      }

      if (i < args.length) {
        simpleArgs.push('...');
      }

      return simpleArgs;
    }

    var simpleObj = {};

    for (var key in args) {
      simpleObj[key] = simplifyArgs(args[key]);
    }

    return simpleObj;
  }

  function Stepper(panel, pageIndex, initialBreakPoints) {
    this.panel = panel;
    this.breakPoint = 0;
    this.nextBreakPoint = null;
    this.pageIndex = pageIndex;
    this.breakPoints = initialBreakPoints;
    this.currentIdx = -1;
    this.operatorListIdx = 0;
  }

  Stepper.prototype = {
    init: function init(operatorList) {
      var panel = this.panel;
      var content = c('div', 'c=continue, s=step');
      var table = c('table');
      content.appendChild(table);
      table.cellSpacing = 0;
      var headerRow = c('tr');
      table.appendChild(headerRow);
      headerRow.appendChild(c('th', 'Break'));
      headerRow.appendChild(c('th', 'Idx'));
      headerRow.appendChild(c('th', 'fn'));
      headerRow.appendChild(c('th', 'args'));
      panel.appendChild(content);
      this.table = table;
      this.updateOperatorList(operatorList);
    },
    updateOperatorList: function updateOperatorList(operatorList) {
      var self = this;

      function cboxOnClick() {
        var x = +this.dataset.idx;

        if (this.checked) {
          self.breakPoints.push(x);
        } else {
          self.breakPoints.splice(self.breakPoints.indexOf(x), 1);
        }

        StepperManager.saveBreakPoints(self.pageIndex, self.breakPoints);
      }

      var MAX_OPERATORS_COUNT = 15000;

      if (this.operatorListIdx > MAX_OPERATORS_COUNT) {
        return;
      }

      var chunk = document.createDocumentFragment();
      var operatorsToDisplay = Math.min(MAX_OPERATORS_COUNT, operatorList.fnArray.length);

      for (var i = this.operatorListIdx; i < operatorsToDisplay; i++) {
        var line = c('tr');
        line.className = 'line';
        line.dataset.idx = i;
        chunk.appendChild(line);
        var checked = this.breakPoints.includes(i);
        var args = operatorList.argsArray[i] || [];
        var breakCell = c('td');
        var cbox = c('input');
        cbox.type = 'checkbox';
        cbox.className = 'points';
        cbox.checked = checked;
        cbox.dataset.idx = i;
        cbox.onclick = cboxOnClick;
        breakCell.appendChild(cbox);
        line.appendChild(breakCell);
        line.appendChild(c('td', i.toString()));
        var fn = opMap[operatorList.fnArray[i]];
        var decArgs = args;

        if (fn === 'showText') {
          var glyphs = args[0];
          var newArgs = [];
          var str = [];

          for (var j = 0; j < glyphs.length; j++) {
            var glyph = glyphs[j];

            if (_typeof(glyph) === 'object' && glyph !== null) {
              str.push(glyph.fontChar);
            } else {
              if (str.length > 0) {
                newArgs.push(str.join(''));
                str = [];
              }

              newArgs.push(glyph);
            }
          }

          if (str.length > 0) {
            newArgs.push(str.join(''));
          }

          decArgs = [newArgs];
        }

        line.appendChild(c('td', fn));
        line.appendChild(c('td', JSON.stringify(simplifyArgs(decArgs))));
      }

      if (operatorsToDisplay < operatorList.fnArray.length) {
        line = c('tr');
        var lastCell = c('td', '...');
        lastCell.colspan = 4;
        chunk.appendChild(lastCell);
      }

      this.operatorListIdx = operatorList.fnArray.length;
      this.table.appendChild(chunk);
    },
    getNextBreakPoint: function getNextBreakPoint() {
      this.breakPoints.sort(function (a, b) {
        return a - b;
      });

      for (var i = 0; i < this.breakPoints.length; i++) {
        if (this.breakPoints[i] > this.currentIdx) {
          return this.breakPoints[i];
        }
      }

      return null;
    },
    breakIt: function breakIt(idx, callback) {
      StepperManager.selectStepper(this.pageIndex, true);
      var self = this;
      var dom = document;
      self.currentIdx = idx;

      var listener = function listener(e) {
        switch (e.keyCode) {
          case 83:
            dom.removeEventListener('keydown', listener);
            self.nextBreakPoint = self.currentIdx + 1;
            self.goTo(-1);
            callback();
            break;

          case 67:
            dom.removeEventListener('keydown', listener);
            var breakPoint = self.getNextBreakPoint();
            self.nextBreakPoint = breakPoint;
            self.goTo(-1);
            callback();
            break;
        }
      };

      dom.addEventListener('keydown', listener);
      self.goTo(idx);
    },
    goTo: function goTo(idx) {
      var allRows = this.panel.getElementsByClassName('line');

      for (var x = 0, xx = allRows.length; x < xx; ++x) {
        var row = allRows[x];

        if ((row.dataset.idx | 0) === idx) {
          row.style.backgroundColor = 'rgb(251,250,207)';
          row.scrollIntoView();
        } else {
          row.style.backgroundColor = null;
        }
      }
    }
  };
  return Stepper;
}();

var Stats = function Stats() {
  var stats = [];

  function clear(node) {
    while (node.hasChildNodes()) {
      node.removeChild(node.lastChild);
    }
  }

  function getStatIndex(pageNumber) {
    for (var i = 0, ii = stats.length; i < ii; ++i) {
      if (stats[i].pageNumber === pageNumber) {
        return i;
      }
    }

    return false;
  }

  return {
    id: 'Stats',
    name: 'Stats',
    panel: null,
    manager: null,
    init: function init(pdfjsLib) {
      this.panel.setAttribute('style', 'padding: 5px;');
    },
    enabled: false,
    active: false,
    add: function add(pageNumber, stat) {
      if (!stat) {
        return;
      }

      var statsIndex = getStatIndex(pageNumber);

      if (statsIndex !== false) {
        var b = stats[statsIndex];
        this.panel.removeChild(b.div);
        stats.splice(statsIndex, 1);
      }

      var wrapper = document.createElement('div');
      wrapper.className = 'stats';
      var title = document.createElement('div');
      title.className = 'title';
      title.textContent = 'Page: ' + pageNumber;
      var statsDiv = document.createElement('div');
      statsDiv.textContent = stat.toString();
      wrapper.appendChild(title);
      wrapper.appendChild(statsDiv);
      stats.push({
        pageNumber: pageNumber,
        div: wrapper
      });
      stats.sort(function (a, b) {
        return a.pageNumber - b.pageNumber;
      });
      clear(this.panel);

      for (var i = 0, ii = stats.length; i < ii; ++i) {
        this.panel.appendChild(stats[i].div);
      }
    },
    cleanup: function cleanup() {
      stats = [];
      clear(this.panel);
    }
  };
}();

window.PDFBug = function PDFBugClosure() {
  var panelWidth = 300;
  var buttons = [];
  var activePanel = null;
  return {
    tools: [FontInspector, StepperManager, Stats],
    enable: function enable(ids) {
      var all = false,
          tools = this.tools;

      if (ids.length === 1 && ids[0] === 'all') {
        all = true;
      }

      for (var i = 0; i < tools.length; ++i) {
        var tool = tools[i];

        if (all || ids.includes(tool.id)) {
          tool.enabled = true;
        }
      }

      if (!all) {
        tools.sort(function (a, b) {
          var indexA = ids.indexOf(a.id);
          indexA = indexA < 0 ? tools.length : indexA;
          var indexB = ids.indexOf(b.id);
          indexB = indexB < 0 ? tools.length : indexB;
          return indexA - indexB;
        });
      }
    },
    init: function init(pdfjsLib, container) {
      var ui = document.createElement('div');
      ui.id = 'PDFBug';
      var controls = document.createElement('div');
      controls.setAttribute('class', 'controls');
      ui.appendChild(controls);
      var panels = document.createElement('div');
      panels.setAttribute('class', 'panels');
      ui.appendChild(panels);
      container.appendChild(ui);
      container.style.right = panelWidth + 'px';
      var tools = this.tools;
      var self = this;

      for (var i = 0; i < tools.length; ++i) {
        var tool = tools[i];
        var panel = document.createElement('div');
        var panelButton = document.createElement('button');
        panelButton.textContent = tool.name;
        panelButton.addEventListener('click', function (selected) {
          return function (event) {
            event.preventDefault();
            self.selectPanel(selected);
          };
        }(i));
        controls.appendChild(panelButton);
        panels.appendChild(panel);
        tool.panel = panel;
        tool.manager = this;

        if (tool.enabled) {
          tool.init(pdfjsLib);
        } else {
          panel.textContent = tool.name + ' is disabled. To enable add ' + ' "' + tool.id + '" to the pdfBug parameter ' + 'and refresh (separate multiple by commas).';
        }

        buttons.push(panelButton);
      }

      this.selectPanel(0);
    },
    cleanup: function cleanup() {
      for (var i = 0, ii = this.tools.length; i < ii; i++) {
        if (this.tools[i].enabled) {
          this.tools[i].cleanup();
        }
      }
    },
    selectPanel: function selectPanel(index) {
      if (typeof index !== 'number') {
        index = this.tools.indexOf(index);
      }

      if (index === activePanel) {
        return;
      }

      activePanel = index;
      var tools = this.tools;

      for (var j = 0; j < tools.length; ++j) {
        if (j === index) {
          buttons[j].setAttribute('class', 'active');
          tools[j].active = true;
          tools[j].panel.removeAttribute('hidden');
        } else {
          buttons[j].setAttribute('class', '');
          tools[j].active = false;
          tools[j].panel.setAttribute('hidden', 'true');
        }
      }
    }
  };
}();