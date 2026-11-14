// Small DOM helpers for the copilot sidebar.
"use strict";

const { setIcon } = require("obsidian");

function clear(element) {
  if (element.empty) element.empty();
  else while (element.firstChild) element.removeChild(element.firstChild);
}

function create(parent, tag, cls, text) {
  const options = {};
  if (cls) options.cls = cls;
  if (text !== undefined) options.text = text;
  if (parent.createEl) return parent.createEl(tag, options);
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  parent.appendChild(node);
  return node;
}

// An Obsidian icon in `el`, where Obsidian is there to draw one.
function icon(el, name) {
  if (typeof setIcon === "function") setIcon(el, name);
  return el;
}

// An Obsidian icon, or `glyph` where none was drawn (no Obsidian, or a name its icon set lacks).
function glyphIcon(el, name, glyph) {
  if (typeof setIcon === "function") setIcon(el, name);
  if (!el.querySelector?.("svg")) el.textContent = glyph;
  return el;
}

function iconButton(parent, icon, label, cls = "clickable-icon") {
  const button = create(parent, "button", cls);
  button.setAttribute("aria-label", label);
  button.setAttribute("title", label);
  if (typeof setIcon === "function") setIcon(button, icon);
  else button.textContent = label;
  return button;
}

// One line until they type more; capped so a long paste does not push the chat off-screen.
function autoGrow(input, max = 180) {
  if (!input || !input.style) return;
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight || 0, max)}px`;
}

function atBottom(element) {
  return element.scrollHeight - element.scrollTop - element.clientHeight <= 40;
}

module.exports = { clear, create, icon, glyphIcon, iconButton, autoGrow, atBottom };
