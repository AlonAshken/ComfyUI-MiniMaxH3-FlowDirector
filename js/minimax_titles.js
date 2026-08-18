const { app } = window.comfyAPI.app;

const RENAMED = {
  MiniMaxH3FlowDirectorCS: "MiniMax H3 Flow Director",
  MiniMaxH3PreviewOverrideCS: "MiniMax H3 Preview Override",
  MiniMaxH3EnhancePromptCS: "MiniMax H3 Enhance Prompt",
  MiniMaxH3SaveLastFrameCS: "MiniMax H3 Save Last Frame",
};

function healTitle(node) {
  const current = RENAMED[node?.type];
  if (!current || !node.title) return;
  if (node.title === current + " CS" || node.title === current + " -CS") {
    node.title = current;
    node.setDirtyCanvas?.(true, true);
  }
}

app.registerExtension({
  name: "MiniMaxH3Flow.TitleCleanup",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!RENAMED[nodeData.name]) return;
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const out = onConfigure?.apply(this, arguments);
      healTitle(this);
      return out;
    };
  },

  async setup() {
    setTimeout(() => (app.graph?._nodes || []).forEach(healTitle), 0);
  },

  async afterConfigureGraph() {
    (app.graph?._nodes || []).forEach(healTitle);
  },
});
