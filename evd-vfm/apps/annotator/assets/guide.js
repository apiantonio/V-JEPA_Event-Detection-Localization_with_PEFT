"use strict";
// One table drives the real keys, the command labels and the Guide.
// Every new feature must also be described in the helpDialog chapters.
const SHORTCUTS = [
  {id:"save",label:"Save: reassign video IDs by default, confirming drafts is optional",keys:["s"],ctrl:true,chord:"Ctrl + S"},
  {id:"confirm",label:"Confirm the review, save and advance",keys:["enter"],ctrl:true,chord:"Ctrl + Enter"},
  {id:"undo",label:"Undo change",keys:["z"],ctrl:true,chord:"Ctrl + Z"},
  {id:"redo",label:"Redo change",keys:["y"],ctrl:true,chord:"Ctrl + Y"},
  {id:"redo",label:"Redo change (alternative)",keys:["z"],ctrl:true,shift:true,chord:"Ctrl + Shift + Z"},
  {id:"all",label:"Select all boxes",keys:["a"],ctrl:true,chord:"Ctrl + A"},
  {id:"copyBoxes",label:"Copy the selected boxes",keys:["c"],ctrl:true,chord:"Ctrl + C"},
  {id:"pasteBoxes",label:"Paste the boxes into the current plane / frame",keys:["v"],ctrl:true,chord:"Ctrl + V"},
  {id:"select",label:"Select / move",keys:["v"],chord:"V"},
  {id:"draw",label:"Draw a box",keys:["b"],chord:"B"},
  {id:"grid",label:"Show / hide the token grid",keys:["g"],chord:"G"},
  {id:"tokens",label:"Show / hide the token-space version",keys:["t"],chord:"T"},
  {id:"precision",label:"Thin borders without fill",keys:["l"],chord:"L"},
  {id:"unitMode",label:"Toggle VTCD planes / source samples and frames",keys:["m"],chord:"M"},
  {id:"copy",label:"Copy from the previous plane",keys:["p"],chord:"P"},
  {id:"applyPlanes",label:"Apply the boxes to several planes",keys:["p"],shift:true,chord:"Shift + P"},
  {id:"original",label:"Show / hide the original",keys:["h"],chord:"H"},
  {id:"previous",label:"Previous plane / sample (frame in Frames mode)",keys:["arrowleft"],chord:"←"},
  {id:"next",label:"Next plane / sample (frame in Frames mode)",keys:["arrowright"],chord:"→"},
  {id:"previous10",label:"Back 10 units",keys:["arrowleft"],shift:true,chord:"Shift + ←"},
  {id:"next10",label:"Forward 10 units",keys:["arrowright"],shift:true,chord:"Shift + →"},
  {id:"previousEvent",label:"Previous event",keys:["["],chord:"["},
  {id:"nextEvent",label:"Next event",keys:["]"],chord:"]"},
  {id:"play",label:"Start / stop preview",keys:[" "],chord:"Space"},
  {id:"move",label:"Move selection by one pixel",keys:["arrowleft","arrowright","arrowup","arrowdown"],alt:true,chord:"Alt + arrows"},
  {id:"move10",label:"Move selection by 10 pixels",keys:["arrowleft","arrowright","arrowup","arrowdown"],alt:true,shift:true,chord:"Alt + Shift + arrows"},
  {id:"zoomIn",label:"Zoom in",keys:["+","="],shift:null,chord:"+ or ="},
  {id:"zoomOut",label:"Zoom out",keys:["-","_"],shift:null,chord:"−"},
  {id:"zoomFit",label:"Fit video to the window",keys:["0"],chord:"0"},
  {id:"delete",label:"Delete selection",keys:["delete","backspace"],chord:"Del / Backspace"},
  {id:"escape",label:"Close menu / cancel gesture / deselect",keys:["escape"],chord:"Esc"},
  {id:"menu",label:"Menu of the selected box",keys:["f10"],shift:true,chord:"Shift + F10"},
  {id:"help",label:"Open Guide",keys:["?"],shift:null,chord:"?"},
];
const shortcutLabel = id => SHORTCUTS.find(s=>s.id===id).chord;
function shortcutAction(event) {
  return SHORTCUTS.find(s=>s.keys.includes(event.key.toLowerCase())
    && !!s.ctrl===(event.ctrlKey||event.metaKey) && !!s.alt===event.altKey
    && (s.shift===null || !!s.shift===event.shiftKey))?.id;
}
function openGuide() {
  stopPlay();closeBoxMenu();$("shortcutList").replaceChildren();
  for(const entry of SHORTCUTS) $("shortcutList").append(el("span",{},entry.label),el("kbd",{},entry.chord));
  $("helpDialog").showModal();
}
