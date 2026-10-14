// The notes editor's Markdown: what a rendered node came from, and back again.
"use strict";

function markdownBlockShortcut(text) {
  const heading = String(text || "").match(/^(#{1,6})\s$/);
  if (heading) return { tag: `h${heading[1].length}`, kind: "block" };
  if (/^[-*+]\s$/.test(text || "")) return { tag: "ul", kind: "list" };
  if (/^\d+\.\s$/.test(text || "")) return { tag: "ol", kind: "list" };
  if (/^>\s$/.test(text || "")) return { tag: "blockquote", kind: "block" };
  return null;
}

/* Markdown this editor RENDERS but cannot reconstruct from the DOM: an Obsidian embed, a
 * formula, a wikilink. Each rendered node keeps the source it came from on `data-slim-src`, and
 * the serializer hands that back verbatim.
 *
 * ⚠ WITHOUT THIS THE EDITOR EATS THEIR SCREENSHOTS. `markdownFromEditable` runs on every
 * keystroke over the RENDERED tree, and `inlineMarkdown` had no case for an <img> — so a note
 * containing `![[Attachments/Recorder/…png]]` lost the embed the moment they typed a character
 * next to it. Same for `$…$`, which came back as bare text with its delimiters gone.
 */
const SOURCE_ATOM = /(!?\[\[[^\]]+\]\]|!\[[^\]]*\]\([^)]*\)|\$\$[^$]+\$\$|\$[^$\n]+\$)/g;

function stampSourceAtoms(root, markdown) {
  if (!root || typeof root.querySelectorAll !== "function") return;
  const atoms = String(markdown || "").match(SOURCE_ATOM) || [];
  if (!atoms.length) return;
  const matched = Array.from(
    root.querySelectorAll("img, .internal-embed, a.internal-link, mjx-container, .math"));
  // ⚠ OUTERMOST ONLY. Obsidian renders an embed as `<span class="internal-embed"><img></span>`
  // and a formula as `<span class="math"><mjx-container>`: two matches for one atom. Counting
  // both put every source after the first on the wrong node and the second half on none, and
  // each blur-save of the notes pane then halved their screenshots — 7, 4, 2 (2026-09-05).
  const inside = new Set(matched);
  const nodes = matched.filter((node) => {
    for (let up = node.parentElement; up; up = up.parentElement) if (inside.has(up)) return false;
    return true;
  });
  for (let i = 0; i < nodes.length && i < atoms.length; i += 1) {
    if (typeof nodes[i].setAttribute === "function") {
      nodes[i].setAttribute("data-slim-src", atoms[i]);
    }
  }
}

function inlineMarkdown(node) {
  if (!node) return "";
  if (node.nodeType === 3) return node.nodeValue || "";
  // The source this node was rendered from, when it is something we cannot rebuild.
  const stamped = typeof node.getAttribute === "function"
    ? node.getAttribute("data-slim-src") : null;
  if (stamped) return stamped;
  const tag = String(node.tagName || "").toLowerCase();
  if (tag === "br") return "\n";
  const inner = Array.from(node.childNodes || [], inlineMarkdown).join("");
  if (tag === "strong" || tag === "b") return `**${inner}**`;
  if (tag === "em" || tag === "i") return `*${inner}*`;
  if (tag === "code") return `\`${inner}\``;
  if (tag === "a") return `[${inner}](${node.getAttribute("href") || ""})`;
  return inner;
}

function markdownFromEditable(root) {
  const list = (node, depth) => {
    const ordered = String(node.tagName || "").toLowerCase() === "ol";
    return Array.from(node.children || []).map((li, i) => {
      // ⚠ An inner list is a CHILD of its <li>. Serializing the item inline flattened it into
      // the item's text, and their sub-bullets came back as a bare paragraph (2026-09-05).
      const own = [], inner = [];
      for (const child of li.childNodes || []) {
        const tag = String(child.tagName || "").toLowerCase();
        if (tag === "ul" || tag === "ol") inner.push(list(child, depth + 1));
        else own.push(inlineMarkdown(child));
      }
      const marker = ordered ? `${i + 1}.` : "-";
      return [`${"\t".repeat(depth)}${marker} ${own.join("").trim()}`, ...inner].join("\n");
    }).join("\n");
  };
  const block = (node) => {
    const tag = String(node.tagName || "").toLowerCase();
    if (/^h[1-6]$/.test(tag)) return `${"#".repeat(Number(tag[1]))} ${inlineMarkdown(node).trim()}`;
    if (tag === "ul" || tag === "ol") return list(node, 0);
    if (tag === "blockquote") {
      return inlineMarkdown(node).split("\n").map((line) => `> ${line}`).join("\n").trim();
    }
    if (tag === "pre") return `\`\`\`\n${node.textContent || ""}\n\`\`\``;
    if (tag === "hr") return "---";
    return inlineMarkdown(node).trimEnd();
  };
  return Array.from((root && root.children) || []).map(block)
    .filter((part) => part.trim()).join("\n\n").trim();
}

module.exports = { markdownBlockShortcut, SOURCE_ATOM, stampSourceAtoms, inlineMarkdown,
        markdownFromEditable };
