// Headless helper for src/tools/malware_tools.py (ghidra_decompile).
//
// Usage (as -postScript arguments):
//   DecompileTargets.java list      <out.json> <api1+api2+...|->
//       writes {"entry": "...", "callers": [{"name","address","apis":[...]}]}
//   DecompileTargets.java decompile <out.json> <addr1+addr2+...> <maxFunctions>
// ("+" separates items because cmd.exe splits .bat arguments on commas.)
//       writes {"functions": [{"name","address","code"} | {"name","address","error"}]}
//
// Written in Java (not Jython/PyGhidra) so it runs on every Ghidra version.
//@category Analysis

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.address.AddressIterator;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
import ghidra.program.model.symbol.Reference;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class DecompileTargets extends GhidraScript {

    @Override
    public void run() throws Exception {
        String[] args = getScriptArgs();
        if (args.length < 2) {
            printerr("usage: list|decompile <out.json> [...]");
            return;
        }
        String mode = args[0];
        String out = args[1];
        String list = args.length > 2 ? args[2] : "";
        if (mode.equals("list")) {
            write(out, listCallers(list));
        } else {
            int max = args.length > 3 ? Integer.parseInt(args[3]) : 5;
            write(out, decompile(list, max));
        }
    }

    private static String stripAW(String name) {
        if (name.length() > 1 && (name.endsWith("A") || name.endsWith("W"))) {
            return name.substring(0, name.length() - 1);
        }
        return name;
    }

    private String entryAddress() {
        AddressIterator it = currentProgram.getSymbolTable().getExternalEntryPointIterator();
        while (it.hasNext()) {
            Address a = it.next();
            Function f = getFunctionAt(a);
            if (f != null) {
                return a.toString();
            }
        }
        Function f = getFirstFunction();
        return f == null ? "" : f.getEntryPoint().toString();
    }

    private String listCallers(String apiList) {
        Set<String> wanted = new HashSet<>();
        for (String s : apiList.split("[+,]")) {
            if (!s.isEmpty() && !s.equals("-")) {
                wanted.add(s);
                wanted.add(stripAW(s));
            }
        }
        Map<Function, Set<String>> callers = new LinkedHashMap<>();
        FunctionIterator ext = currentProgram.getFunctionManager().getExternalFunctions();
        while (ext.hasNext() && !monitor.isCancelled()) {
            Function api = ext.next();
            String name = api.getName();
            if (!wanted.contains(name) && !wanted.contains(stripAW(name))) {
                continue;
            }
            // Walk references: external -> IAT pointer / thunk -> calling code (depth <= 3).
            Deque<Address> queue = new ArrayDeque<>();
            Set<Address> visited = new HashSet<>();
            queue.add(api.getEntryPoint());
            int depth = 0;
            while (!queue.isEmpty() && depth < 3) {
                Deque<Address> next = new ArrayDeque<>();
                for (Address target : queue) {
                    for (Reference ref : getReferencesTo(target)) {
                        Address from = ref.getFromAddress();
                        if (!visited.add(from)) {
                            continue;
                        }
                        Function f = getFunctionContaining(from);
                        if (f == null) {
                            next.add(from); // data pointer (IAT slot): follow references to it
                        } else if (f.isThunk()) {
                            next.add(f.getEntryPoint());
                        } else {
                            callers.computeIfAbsent(f, k -> new LinkedHashSet<>()).add(name);
                        }
                    }
                }
                queue = next;
                depth++;
            }
        }
        StringBuilder sb = new StringBuilder();
        sb.append("{\"entry\": ").append(q(entryAddress())).append(", \"callers\": [");
        boolean first = true;
        for (Map.Entry<Function, Set<String>> e : callers.entrySet()) {
            if (!first) sb.append(", ");
            first = false;
            sb.append("{\"name\": ").append(q(e.getKey().getName()))
              .append(", \"address\": ").append(q(e.getKey().getEntryPoint().toString()))
              .append(", \"apis\": [");
            boolean f2 = true;
            for (String a : e.getValue()) {
                if (!f2) sb.append(", ");
                f2 = false;
                sb.append(q(a));
            }
            sb.append("]}");
        }
        sb.append("]}");
        return sb.toString();
    }

    private String decompile(String addrList, int max) {
        DecompInterface ifc = new DecompInterface();
        ifc.openProgram(currentProgram);
        List<String> items = new ArrayList<>();
        Set<Function> done = new HashSet<>();
        try {
            for (String s : addrList.split("[+,]")) {
                if (s.isEmpty() || done.size() >= max || monitor.isCancelled()) {
                    continue;
                }
                String hex = s.toLowerCase().startsWith("0x") ? s.substring(2) : s;
                Address a = toAddr(hex);
                Function f = a == null ? null : getFunctionContaining(a);
                if (f == null && a != null) {
                    f = getFunctionAt(a);
                }
                if (f == null) {
                    items.add("{\"name\": " + q("?") + ", \"address\": " + q(s)
                              + ", \"error\": " + q("no function at address") + "}");
                    continue;
                }
                if (!done.add(f)) {
                    continue;
                }
                DecompileResults r = ifc.decompileFunction(f, 60, monitor);
                String addr = f.getEntryPoint().toString();
                if (r != null && r.decompileCompleted() && r.getDecompiledFunction() != null) {
                    items.add("{\"name\": " + q(f.getName()) + ", \"address\": " + q(addr)
                              + ", \"code\": " + q(r.getDecompiledFunction().getC()) + "}");
                } else {
                    String err = r == null ? "decompiler returned nothing" : r.getErrorMessage();
                    items.add("{\"name\": " + q(f.getName()) + ", \"address\": " + q(addr)
                              + ", \"error\": " + q(err) + "}");
                }
            }
        } finally {
            ifc.dispose();
        }
        return "{\"functions\": [" + String.join(", ", items) + "]}";
    }

    private static String q(String s) {
        if (s == null) return "null";
        StringBuilder sb = new StringBuilder("\"");
        for (char c : s.toCharArray()) {
            switch (c) {
                case '"': sb.append("\\\""); break;
                case '\\': sb.append("\\\\"); break;
                case '\n': sb.append("\\n"); break;
                case '\r': sb.append("\\r"); break;
                case '\t': sb.append("\\t"); break;
                default:
                    if (c < 0x20) sb.append(String.format("\\u%04x", (int) c));
                    else sb.append(c);
            }
        }
        return sb.append('"').toString();
    }

    private static void write(String path, String json) throws Exception {
        try (Writer w = new OutputStreamWriter(new FileOutputStream(new File(path)), StandardCharsets.UTF_8)) {
            w.write(json);
        }
    }
}
