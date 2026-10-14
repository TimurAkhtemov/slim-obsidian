// The plugin's settings and their defaults.
"use strict";

const { PluginSettingTab, Setting } = require("obsidian");
const os = require("os");

// ⚠ Defaults are DERIVED from the home directory, never a literal path — this file is in a
// git repo that may one day be public, and a hard-coded /Users/<name> is both a leak and a
// machine-specific bug.
const DEFAULTS = {
  port: 7546,
  repoPath: `${os.homedir()}/Programming/Personal/slim-public`,
  uvPath: `${os.homedir()}/.local/bin/uv`,
  recordVoice: true,
  lastType: "meeting-note",
};

class SlimSettingTab extends PluginSettingTab {
  constructor(app, plugin) { super(app, plugin); this.plugin = plugin; }
  display() {
    const { containerEl } = this;
    containerEl.empty();
    new Setting(containerEl)
      .setName("Server port")
      .setDesc("Where `slim chat` listens on 127.0.0.1. A server already on this port is " +
               "reused, never duplicated.")
      .addText((t) => t
        .setValue(String(this.plugin.settings.port))
        .onChange(async (v) => {
          const n = parseInt(v, 10);
          if (!isNaN(n)) { this.plugin.settings.port = n; await this.plugin.saveSettings(); }
        }));

    new Setting(containerEl)
      .setName("SLIM repo")
      .setDesc("The checkout `uv run slim chat` is started from.")
      .addText((t) => t
        .setValue(this.plugin.settings.repoPath)
        .onChange(async (v) => {
          this.plugin.settings.repoPath = v.trim();
          await this.plugin.saveSettings();
        }));

    new Setting(containerEl)
      .setName("uv binary")
      .setDesc("Full path — Obsidian launches with a bare PATH, so a bare `uv` will not resolve.")
      .addText((t) => t
        .setValue(this.plugin.settings.uvPath)
        .onChange(async (v) => {
          this.plugin.settings.uvPath = v.trim();
          await this.plugin.saveSettings();
        }));

    containerEl.createEl("p", { cls: "slim-hint",
      text: "Changes take effect the next time the server is started — disable and re-enable " +
            "the plugin to restart it now." });
  }
}

module.exports = { DEFAULTS, SlimSettingTab };
