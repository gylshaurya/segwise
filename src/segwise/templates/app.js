// segwise shared behaviour: tabs, copy buttons, sortable/filterable tables. No dependencies.
(function () {
  document.addEventListener("click", function (e) {
    var tab = e.target.closest("[data-tab]");
    if (tab) {
      var bar = tab.parentElement, sec = bar.parentElement;
      bar.querySelectorAll("[data-tab]").forEach(function (b) { b.classList.toggle("on", b === tab); });
      sec.querySelectorAll("[data-pane]").forEach(function (p) { p.classList.toggle("hide", p.dataset.pane !== tab.dataset.tab); });
    }
    var cp = e.target.closest("[data-copy]");
    if (cp) {
      var txt = cp.parentElement.textContent.replace(cp.textContent, "").trim();
      var lab = cp.querySelector("span");
      (navigator.clipboard ? navigator.clipboard.writeText(txt) : Promise.reject()).then(function () {
        if (lab) { lab.textContent = " Copied"; setTimeout(function () { lab.textContent = " Copy"; }, 1400); }
      }).catch(function () {});
    }
    var th = e.target.closest("th.th-sort");
    if (th) {
      var table = th.closest("table"), idx = Array.prototype.indexOf.call(th.parentElement.children, th);
      var dir = th.dataset.dir === "asc" ? "desc" : "asc"; th.dataset.dir = dir;
      var rows = Array.prototype.slice.call(table.tBodies[0].rows);
      rows.sort(function (a, b) {
        var x = a.cells[idx].dataset.v || a.cells[idx].textContent.trim(), y = b.cells[idx].dataset.v || b.cells[idx].textContent.trim();
        var nx = parseFloat(x), ny = parseFloat(y);
        var c = (!isNaN(nx) && !isNaN(ny)) ? nx - ny : x.localeCompare(y);
        return dir === "asc" ? c : -c;
      });
      rows.forEach(function (r) { table.tBodies[0].appendChild(r); });
    }
    var f = e.target.closest("[data-filter]");
    if (f) {
      var want = f.dataset.filter;
      document.querySelectorAll("[data-filter]").forEach(function (x) { x.classList.toggle("on", x === f); });
      document.querySelectorAll("tr[data-status]").forEach(function (r) { r.classList.toggle("hide", want !== "all" && r.dataset.status !== want); });
    }
  });
  var q = document.querySelector("[data-search]");
  if (q) q.addEventListener("input", function () {
    var s = q.value.toLowerCase();
    document.querySelectorAll(q.dataset.search).forEach(function (el) { el.classList.toggle("hide", s && el.textContent.toLowerCase().indexOf(s) < 0); });
  });
})();
