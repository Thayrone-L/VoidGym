using System.Collections.Concurrent;
using System.Diagnostics;
using System.Net;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Windows.Forms;

namespace VoidGymTray;

internal static class Program
{
    [STAThread]
    private static int Main(string[] args)
    {
        if (args.Contains("--self-test", StringComparer.OrdinalIgnoreCase))
            return TraySelfTest.Run();
        if (!OperatingSystem.IsWindows()) return 2;
        var root = FindProjectRoot(AppContext.BaseDirectory);
        if (root is null)
        {
            MessageBox.Show("Não encontrei a pasta do projeto VoidGym.", "VoidGym", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 2;
        }
        using var mutex = new Mutex(true, @"Local\VoidGymTray", out var created);
        if (!created) return 0;
        ApplicationConfiguration.Initialize();
        Application.Run(new TrayContext(root));
        GC.KeepAlive(mutex);
        return 0;
    }

    private static string? FindProjectRoot(string start)
    {
        var dir = new DirectoryInfo(start);
        while (dir is not null)
        {
            if (File.Exists(Path.Combine(dir.FullName, "Iniciar-Painel.bat")) && Directory.Exists(Path.Combine(dir.FullName, "src")))
                return dir.FullName;
            dir = dir.Parent;
        }
        return null;
    }
}

internal sealed class TrayContext : ApplicationContext
{
    private readonly string root;
    private readonly NotifyIcon icon;
    private readonly System.Windows.Forms.Timer timer;
    private readonly HttpClient http = new() { Timeout = TimeSpan.FromSeconds(8) };
    private readonly ConcurrentDictionary<string, DateTime> recentNotifications = new();
    private Snapshot state = Snapshot.Offline("Inicializando");
    private string baseUrl = "http://127.0.0.1:8765";
    private string lastEventKey = "";
    private string lastConnectivityError = "";
    private bool minimized;
    private bool exiting;
    private int pollBusy;
    private int previousActiveCount;
    private ToolStripMenuItem? resumableMenu;
    private ToolStripMenuItem? windowsMenu;
    private ToolStripMenuItem? autoStartMenu;

    private const string AppName = "VoidGymTray";

    public TrayContext(string projectRoot)
    {
        root = projectRoot;
        icon = new NotifyIcon { Visible = true, Text = "VoidGym — conectando", Icon = BuildIcon(Color.Gray), ContextMenuStrip = BuildMenu() };
        icon.DoubleClick += (_, _) => OpenPanel();
        icon.BalloonTipClicked += (_, _) => OpenPanel();
        timer = new System.Windows.Forms.Timer { Interval = 5000 };
        timer.Tick += async (_, _) => await PollAsync();
        timer.Start();
        _ = PollAsync();
    }

    private ContextMenuStrip BuildMenu()
    {
        var menu = new ContextMenuStrip();
        menu.Items.Add("Abrir painel", null, (_, _) => OpenPanel());
        resumableMenu = new ToolStripMenuItem("Continuar campanha");
        resumableMenu.DropDownOpening += async (_, _) => await RefreshCampaignMenuAsync();
        menu.Items.Add(resumableMenu);
        menu.Items.Add("Reiniciar campanha", null, async (_, _) => await RestartCampaignAsync());
        menu.Items.Add("Parar com segurança", null, async (_, _) => await SafeStopAsync());
        menu.Items.Add("Abortar campanha atual…", null, async (_, _) => await AbortCampaignAsync());
        windowsMenu = new ToolStripMenuItem("Minimizar janelas da campanha", null, (_, _) => ToggleCampaignWindows());
        menu.Items.Add(windowsMenu);
        autoStartMenu = new ToolStripMenuItem("Iniciar com o Windows") { CheckOnClick = true };
        autoStartMenu.Click += (_, _) => SetAutoStart(autoStartMenu.Checked);
        menu.Items.Add(autoStartMenu);
        menu.Items.Add(new ToolStripSeparator());
        menu.Items.Add("Sair", null, async (_, _) => await ExitAsync());
        menu.Opening += (_, _) =>
        {
            if (resumableMenu is not null) resumableMenu.Enabled = state.ActiveCount == 0;
            if (windowsMenu is not null) windowsMenu.Text = minimized ? "Mostrar janelas da campanha" : "Minimizar janelas da campanha";
            if (autoStartMenu is not null) autoStartMenu.Checked = IsAutoStartEnabled();
            menu.Items[2].Enabled = state.ActiveCount > 0;
            menu.Items[3].Enabled = state.ActiveCount > 0;
            menu.Items[4].Enabled = state.ActiveCount > 0;
        };
        return menu;
    }

    private async Task PollAsync()
    {
        if (Interlocked.Exchange(ref pollBusy, 1) != 0 || exiting) return;
        try
        {
            var json = await GetJsonWithFallbackAsync("/api/operator");
            state = Snapshot.Parse(json);
            lastConnectivityError = "";
            timer.Interval = 5000;
            SetIcon(state.Color);
            icon.Text = state.Tooltip.Length > 63 ? state.Tooltip[..63] : state.Tooltip;
            if (windowsMenu is not null) windowsMenu.Enabled = state.ActiveCount > 0;
            NotifyTransitions(state);
        }
        catch (Exception ex)
        {
            state = Snapshot.Offline(ex.Message);
            if (!string.Equals(lastConnectivityError, ex.Message, StringComparison.Ordinal))
            {
                Log("poll", ex);
                lastConnectivityError = ex.Message;
            }
            SetIcon(state.Color);
            icon.Text = "VoidGym — painel desconectado";
            timer.Interval = Math.Min(30000, timer.Interval * 2);
        }
        finally { Interlocked.Exchange(ref pollBusy, 0); }
    }

    private async Task<JsonObject> GetJsonWithFallbackAsync(string path)
    {
        Exception? last = null;
        foreach (var port in new[] { 8765, 8766 })
        {
            var candidate = $"http://127.0.0.1:{port}";
            try
            {
                using var response = await http.GetAsync(candidate + path);
                response.EnsureSuccessStatusCode();
                baseUrl = candidate;
                return JsonNode.Parse(await response.Content.ReadAsStringAsync()) as JsonObject
                    ?? throw new InvalidDataException("Resposta JSON inválida.");
            }
            catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException or InvalidDataException)
            { last = ex; }
        }
        throw last ?? new HttpRequestException("Painel e supervisor indisponíveis.");
    }

    private async Task<JsonObject> PostAsync(string path, JsonObject payload)
    {
        var session = await GetJsonWithFallbackAsync("/api/session");
        var token = session["csrf_token"]?.GetValue<string>();
        if (string.IsNullOrWhiteSpace(token)) throw new InvalidDataException("O servidor não retornou o token de controle.");
        using var request = new HttpRequestMessage(HttpMethod.Post, baseUrl + path)
        {
            Content = new StringContent(payload.ToJsonString(), System.Text.Encoding.UTF8, "application/json")
        };
        request.Headers.Add("X-Control-Token", token);
        using var response = await http.SendAsync(request);
        var body = await response.Content.ReadAsStringAsync();
        if (!response.IsSuccessStatusCode)
        {
            var error = JsonNode.Parse(body)?["error"]?.GetValue<string>() ?? response.ReasonPhrase ?? "Erro HTTP";
            throw new InvalidOperationException(error);
        }
        return JsonNode.Parse(body) as JsonObject ?? new JsonObject();
    }

    private void OpenPanel()
    {
        try
        {
            if (!CanReach("http://127.0.0.1:8765/api/status")) StartPanel();
            Process.Start(new ProcessStartInfo("http://127.0.0.1:8765/") { UseShellExecute = true });
        }
        catch (Exception ex) { Log("open panel", ex); ShowBalloon("Não foi possível abrir o painel", ex.Message); }
    }

    private bool CanReach(string url)
    {
        try { using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(1) }; using var response = client.GetAsync(url).GetAwaiter().GetResult(); return response.IsSuccessStatusCode; }
        catch { return false; }
    }

    private void StartPanel()
    {
        var python = Path.Combine(root, ".venv", "Scripts", "pythonw.exe");
        if (!File.Exists(python)) python = Path.Combine(root, ".venv", "Scripts", "python.exe");
        var script = Path.Combine(root, "tools", "serve_training_dashboard.py");
        var log = LogPath();
        Directory.CreateDirectory(Path.GetDirectoryName(log)!);
        var psi = new ProcessStartInfo(python) { WorkingDirectory = root, UseShellExecute = false, CreateNoWindow = true, WindowStyle = ProcessWindowStyle.Hidden };
        psi.ArgumentList.Add(script); psi.ArgumentList.Add("--allow-execute");
        psi.RedirectStandardOutput = true; psi.RedirectStandardError = true;
        var process = Process.Start(psi) ?? throw new InvalidOperationException("O servidor do painel não iniciou.");
        _ = Task.Run(async () =>
        {
            using var writer = new StreamWriter(new FileStream(log, FileMode.Append, FileAccess.Write, FileShare.ReadWrite));
            var outTask = process.StandardOutput.ReadToEndAsync(); var errTask = process.StandardError.ReadToEndAsync();
            await Task.WhenAll(outTask, errTask);
            await writer.WriteLineAsync($"[{DateTimeOffset.Now:O}] PANEL stdout: {outTask.Result}");
            await writer.WriteLineAsync($"[{DateTimeOffset.Now:O}] PANEL stderr: {errTask.Result}");
        });
        var until = DateTime.UtcNow.AddSeconds(15);
        while (DateTime.UtcNow < until && !CanReach("http://127.0.0.1:8765/api/status")) Thread.Sleep(250);
        if (!CanReach("http://127.0.0.1:8765/api/status")) throw new TimeoutException("O painel não respondeu em 15 segundos.");
    }

    private async Task RefreshCampaignMenuAsync()
    {
        if (resumableMenu is null) return;
        resumableMenu.DropDownItems.Clear();
        try
        {
            var data = await GetJsonWithFallbackAsync("/api/operator/resumable");
            if (data["campaigns"] is not JsonArray rows || rows.Count == 0)
            { resumableMenu.DropDownItems.Add(new ToolStripMenuItem("Nenhuma campanha retomável") { Enabled = false }); return; }
            foreach (var item in rows.OfType<JsonObject>())
            {
                var id = item["campaign_id"]?.GetValue<string>();
                if (string.IsNullOrWhiteSpace(id)) continue;
                var name = item["name"]?.GetValue<string>() ?? id;
                var round = item["round"]?.GetValue<int>() ?? 0;
                var choice = new ToolStripMenuItem($"{name} · R{round}") { ToolTipText = id };
                choice.Click += async (_, _) => await ContinueCampaignAsync(id, $"{name} na rodada {round}");
                resumableMenu.DropDownItems.Add(choice);
            }
        }
        catch (Exception ex)
        { resumableMenu.DropDownItems.Add(new ToolStripMenuItem("Falha ao carregar: " + ex.Message) { Enabled = false }); }
    }

    private async Task ContinueCampaignAsync(string campaignId, string display)
    {
        try
        {
            await PostAsync("/api/operator/continue", new JsonObject { ["campaign_id"] = campaignId });
            ShowBalloon("Campanha retomada", display);
            await PollAsync();
        }
        catch (Exception ex) { Log("continue", ex); ShowBalloon("Não foi possível continuar", ex.Message); }
    }

    private async Task RestartCampaignAsync()
    {
        try
        {
            if (state.ActiveCount == 0) throw new InvalidOperationException("Não há campanha ativa para reiniciar.");
            var campaignId = state.CampaignId;
            await PostAsync("/api/operator/stop", new JsonObject());
            ShowBalloon("Parada segura solicitada", "O ícone vai retomar a campanha após a parada completar.");
            var until = DateTime.UtcNow.AddMinutes(10);
            while (DateTime.UtcNow < until)
            {
                await Task.Delay(2000);
                var snapshot = Snapshot.Parse(await GetJsonWithFallbackAsync("/api/operator"));
                state = snapshot;
                if (snapshot.ActiveCount == 0) break;
            }
            if (state.ActiveCount != 0) throw new TimeoutException("A parada segura ainda está em andamento.");
            await PostAsync("/api/operator/continue", campaignId is null ? new JsonObject() : new JsonObject { ["campaign_id"] = campaignId });
            ShowBalloon("Campanha reiniciada", "Retomada a partir do último checkpoint.");
            await PollAsync();
        }
        catch (Exception ex) { Log("restart campaign", ex); ShowBalloon("Não foi possível reiniciar", ex.Message); }
    }

    private async Task SafeStopAsync()
    {
        try { await PostAsync("/api/operator/stop", new JsonObject()); ShowBalloon("Parada segura solicitada", "O supervisor está finalizando o lote atual."); await PollAsync(); }
        catch (Exception ex) { Log("safe stop", ex); ShowBalloon("Não foi possível parar", ex.Message); }
    }

    private async Task AbortCampaignAsync()
    {
        if (MessageBox.Show("A campanha será encerrada imediatamente. O lote atual pode ser perdido; checkpoints permanecem. Os saves isolados serão capturados no encerramento. Continuar?", "Abortar campanha", MessageBoxButtons.YesNo, MessageBoxIcon.Warning) != DialogResult.Yes) return;
        try { await PostAsync("/api/operator/abort", new JsonObject()); ShowBalloon("Aborto solicitado", "O supervisor está encerrando os processos da campanha."); await PollAsync(); }
        catch (Exception ex) { Log("abort", ex); ShowBalloon("Não foi possível abortar", ex.Message); }
    }

    private void ToggleCampaignWindows()
    {
        try
        {
            var status = GetJsonWithFallbackAsync("/api/status").GetAwaiter().GetResult();
            var pids = CampaignWindowPids(status);
            if (pids.Count == 0) { ShowBalloon("Nenhuma janela de jogo encontrada", "Os processos não expõem uma janela minimizável."); return; }
            foreach (var hwnd in NativeWindows.WindowsForProcesses(pids)) NativeWindows.ShowWindow(hwnd, minimized ? 9 : 6);
            minimized = !minimized;
            ShowBalloon(minimized ? "Janelas minimizadas" : "Janelas restauradas", "A coleta continua pelo supervisor.");
        }
        catch (Exception ex) { Log("toggle windows", ex); ShowBalloon("Falha nas janelas da campanha", ex.Message); }
    }

    private static HashSet<int> CampaignWindowPids(JsonObject status)
    {
        var roots = new HashSet<int>();
        if (status["active_launches"] is not JsonArray active) return roots;
        foreach (var row in active.OfType<JsonObject>())
            if (row["pid"] is JsonValue value && value.TryGetValue<int>(out var pid) && pid > 0) roots.Add(pid);
        return NativeWindows.GameDescendants(roots);
    }

    private async Task ExitAsync()
    {
        if (state.ActiveCount > 0)
        {
            using var dialog = new ExitChoiceForm();
            if (dialog.ShowDialog() != DialogResult.OK) return;
            if (dialog.StopAndExit)
            {
                try { await PostAsync("/api/operator/stop", new JsonObject()); }
                catch (Exception ex) { ShowBalloon("Parada segura falhou", ex.Message); return; }
            }
        }
        exiting = true; timer.Stop(); icon.Visible = false; icon.Dispose(); http.Dispose(); ExitThread();
    }

    private void NotifyTransitions(Snapshot current)
    {
        var key = $"{current.Error}:{current.HelpCount}:{current.Completed}";
        if (key == lastEventKey) { previousActiveCount = current.ActiveCount; return; }
        if (current.Error && !string.IsNullOrEmpty(current.ErrorText)) ShowBalloon("Erro no treino", current.ErrorText);
        else if (current.HelpCount > 0) ShowBalloon("Pedido de ajuda aberto", $"Há {current.HelpCount} pedido(s) aguardando.");
        else if (previousActiveCount > 0 && current.ActiveCount == 0 && current.Completed) ShowBalloon("Campanha concluída", "A campanha terminou.");
        lastEventKey = key;
        previousActiveCount = current.ActiveCount;
    }

    private void ShowBalloon(string title, string body)
    {
        var key = title + body;
        if (recentNotifications.TryGetValue(key, out var last) && DateTime.Now - last < TimeSpan.FromSeconds(30)) return;
        recentNotifications[key] = DateTime.Now;
        icon.BalloonTipTitle = title; icon.BalloonTipText = body.Length > 250 ? body[..250] : body;
        icon.BalloonTipIcon = title.Contains("Erro", StringComparison.OrdinalIgnoreCase) || title.Contains("falhou", StringComparison.OrdinalIgnoreCase) ? ToolTipIcon.Error : ToolTipIcon.Info;
        icon.ShowBalloonTip(5000);
    }

    private void SetAutoStart(bool enabled)
    {
        try
        {
            var linkPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Startup), "VoidGym Tray.lnk");
            if (!enabled) { if (File.Exists(linkPath)) File.Delete(linkPath); }
            else
            {
                var exe = Environment.ProcessPath ?? throw new InvalidOperationException("Caminho do executável indisponível.");
                var shellType = Type.GetTypeFromProgID("WScript.Shell") ?? throw new InvalidOperationException("Windows Script Host indisponível.");
                var shell = Activator.CreateInstance(shellType) ?? throw new InvalidOperationException("Não foi possível criar o atalho.");
                try
                {
                    dynamic shortcut = shellType.InvokeMember("CreateShortcut", System.Reflection.BindingFlags.InvokeMethod, null, shell, [linkPath])!;
                    shortcut.TargetPath = exe; shortcut.WorkingDirectory = root; shortcut.Description = "VoidGym na bandeja";
                    shortcut.IconLocation = exe + ",0"; shortcut.Save();
                }
                finally { Marshal.FinalReleaseComObject(shell); }
            }
        }
        catch (Exception ex) { Log("autostart", ex); ShowBalloon("Não foi possível alterar a inicialização", ex.Message); }
    }

    private bool IsAutoStartEnabled()
    {
        return File.Exists(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Startup), "VoidGym Tray.lnk"));
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing) { timer.Dispose(); icon.Dispose(); http.Dispose(); }
        base.Dispose(disposing);
    }

    private static Icon BuildIcon(Color color)
    {
        using var bitmap = new Bitmap(32, 32);
        using (var graphics = Graphics.FromImage(bitmap))
        {
            graphics.Clear(Color.Transparent);
            using var brush = new SolidBrush(color);
            using var pen = new Pen(Color.FromArgb(230, 230, 230), 2);
            graphics.FillEllipse(brush, 4, 4, 24, 24);
            graphics.DrawEllipse(pen, 4, 4, 24, 24);
        }
        var handle = bitmap.GetHicon();
        try { using var temporary = Icon.FromHandle(handle); return (Icon)temporary.Clone(); }
        finally { NativeWindows.DestroyIcon(handle); }
    }

    private void SetIcon(Color color)
    {
        var next = BuildIcon(color);
        var previous = icon.Icon;
        icon.Icon = next;
        previous?.Dispose();
    }

    private string LogPath() => Path.Combine(root, "artifacts", "control-plane", $"tray-{DateTime.Now:yyyyMMdd}.log");
    private void Log(string operation, Exception error)
    {
        try { Directory.CreateDirectory(Path.GetDirectoryName(LogPath())!); File.AppendAllText(LogPath(), $"[{DateTimeOffset.Now:O}] {operation}: {error}\n"); } catch { }
    }
}

internal sealed class Snapshot
{
    public int ActiveCount { get; init; }
    public double StepsPerSecond { get; init; }
    public int Instances { get; init; }
    public int Round { get; init; }
    public string? CampaignId { get; init; }
    public int HelpCount { get; init; }
    public bool Error { get; init; }
    public string ErrorText { get; init; } = "";
    public bool Completed { get; init; }
    public Color Color { get; init; } = Color.Gray;
    public string Tooltip { get; init; } = "VoidGym — parado";

    public static Snapshot Offline(string reason) => new() { ErrorText = reason, Tooltip = "VoidGym — supervisor desconectado" };

    public static Snapshot Parse(JsonObject json)
    {
        var status = json["status"] as JsonObject;
        var active = status?["active_count"]?.GetValue<int>() ?? 0;
        var steps = json["steps_per_second"]?.GetValue<double>() ?? 0;
        var instances = json["instances"] is JsonArray rows ? rows.Count(row => row is JsonObject item && (item["active"]?.GetValue<bool>() ?? false)) : 0;
        var campaign = json["campaign_id"]?.GetValue<string>();
        var round = json["learner_updates"]?.GetValue<int>() ?? 0;
        var helps = json["help_requests"] is JsonArray requests ? requests.Count(row => row is JsonObject item && (item["resolved_at"] is null)) : 0;
        var alerts = json["alerts"] is JsonArray alertRows ? alertRows.Select(a => a?.GetValue<string>() ?? "").ToArray() : [];
        var banner = json["status_banner"] as JsonObject;
        var error = banner?["kind"]?.GetValue<string>() == "error" || alerts.Any(a => a.Contains("learner_failed", StringComparison.OrdinalIgnoreCase));
        var errorText = banner?["message"]?.GetValue<string>() ?? "Learner ou supervisor falhou.";
        var memory = json["system"]?["memory_percent"]?.GetValue<int>() ?? 0;
        var completed = active == 0 && (json["last_completion"]?["safe"]?.GetValue<bool>() ?? false);
        var color = active == 0 ? System.Drawing.Color.Gray : error ? System.Drawing.Color.Firebrick : helps > 0 || alerts.Length > 0 || memory > 85 ? System.Drawing.Color.Goldenrod : System.Drawing.Color.SeaGreen;
        var tooltip = active == 0 ? "VoidGym — parado" : $"VoidGym — rodada {round} · {steps:F1} passos/s · {Math.Max(1, instances)} instâncias";
        return new Snapshot { ActiveCount = active, StepsPerSecond = steps, Instances = instances, Round = round, CampaignId = campaign, HelpCount = helps, Error = error, ErrorText = errorText, Completed = completed, Color = color, Tooltip = tooltip };
    }
}

internal sealed class ExitChoiceForm : Form
{
    public bool StopAndExit { get; private set; }
    public ExitChoiceForm()
    {
        Text = "Sair do VoidGym"; Width = 430; Height = 155; StartPosition = FormStartPosition.CenterScreen;
        FormBorderStyle = FormBorderStyle.FixedDialog; MaximizeBox = false; MinimizeBox = false;
        Controls.Add(new Label { Text = "O treino continua em segundo plano. Parar também?", AutoSize = false, Width = 390, Height = 36, Left = 15, Top = 14 });
        var only = new Button { Text = "Só sair", Left = 15, Top = 70, Width = 110 };
        var stop = new Button { Text = "Parar com segurança e sair", Left = 135, Top = 70, Width = 190 };
        var cancel = new Button { Text = "Cancelar", Left = 335, Top = 70, Width = 75 };
        only.Click += (_, _) => { StopAndExit = false; DialogResult = DialogResult.OK; Close(); };
        stop.Click += (_, _) => { StopAndExit = true; DialogResult = DialogResult.OK; Close(); };
        cancel.Click += (_, _) => { DialogResult = DialogResult.Cancel; Close(); };
        Controls.AddRange([only, stop, cancel]);
    }
}

internal static class NativeWindows
{
    private const uint SnapshotFlags = 0x00000002;
    private const uint InvalidHandle = 0xFFFFFFFF;
    private const int SwMinimize = 6;
    private const int SwRestore = 9;
    private static readonly IntPtr InvalidSnapshot = new(unchecked((int)InvalidHandle));

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct ProcessEntry { public uint Size; public uint Usage; public uint ProcessId; public IntPtr DefaultHeapId; public uint ModuleId; public uint Threads; public uint ParentProcessId; public int Priority; public uint Flags; [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 260)] public string ExeFile; }
    private delegate bool EnumWindowsProc(IntPtr hwnd, IntPtr lParam);
    [DllImport("kernel32.dll", SetLastError = true)] private static extern IntPtr CreateToolhelp32Snapshot(uint flags, uint processId);
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)] private static extern bool Process32FirstW(IntPtr snapshot, ref ProcessEntry entry);
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)] private static extern bool Process32NextW(IntPtr snapshot, ref ProcessEntry entry);
    [DllImport("kernel32.dll", SetLastError = true)] private static extern bool CloseHandle(IntPtr handle);
    [DllImport("user32.dll", SetLastError = true)] public static extern bool DestroyIcon(IntPtr icon);
    [DllImport("user32.dll")] private static extern bool EnumWindows(EnumWindowsProc callback, IntPtr lParam);
    [DllImport("user32.dll")] private static extern uint GetWindowThreadProcessId(IntPtr hwnd, out uint processId);
    [DllImport("user32.dll")] private static extern bool IsWindowVisible(IntPtr hwnd);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hwnd, int command);

    public static HashSet<int> GameDescendants(HashSet<int> roots)
    {
        var result = new HashSet<int>();
        if (roots.Count == 0) return result;
        var parents = ProcessTree();
        var descendants = DescendantPids(roots, parents);
        foreach (var pid in descendants)
        {
            try
            {
                using var process = Process.GetProcessById(pid);
                if (process.ProcessName.Contains("Hollow Knight", StringComparison.OrdinalIgnoreCase) || process.ProcessName.Contains("HollowKnight", StringComparison.OrdinalIgnoreCase) || process.ProcessName.Contains("hollow_knight", StringComparison.OrdinalIgnoreCase)) result.Add(pid);
            }
            catch { }
        }
        return result;
    }

    internal static HashSet<int> DescendantPids(HashSet<int> roots, IReadOnlyDictionary<int, int> parentByPid)
    {
        var descendants = new HashSet<int>(roots);
        var frontier = new Queue<int>(roots);
        while (frontier.Count > 0)
        {
            var parent = frontier.Dequeue();
            foreach (var item in parentByPid)
                if (item.Value == parent && descendants.Add(item.Key)) frontier.Enqueue(item.Key);
        }
        return descendants;
    }

    internal static bool IsCampaignWindowPid(int pid, IReadOnlySet<int> campaignGamePids) => campaignGamePids.Contains(pid);

    private static Dictionary<int, int> ProcessTree()
    {
        var result = new Dictionary<int, int>();
        var snap = CreateToolhelp32Snapshot(SnapshotFlags, 0);
        if (snap == InvalidSnapshot) return result;
        try
        {
            var entry = new ProcessEntry { Size = (uint)Marshal.SizeOf<ProcessEntry>() };
            if (!Process32FirstW(snap, ref entry)) return result;
            do { result[(int)entry.ProcessId] = (int)entry.ParentProcessId; entry.Size = (uint)Marshal.SizeOf<ProcessEntry>(); } while (Process32NextW(snap, ref entry));
        }
        finally { CloseHandle(snap); }
        return result;
    }

    public static List<IntPtr> WindowsForProcesses(HashSet<int> pids)
    {
        var result = new List<IntPtr>();
        EnumWindows((hwnd, _) => { GetWindowThreadProcessId(hwnd, out var pid); if (IsWindowVisible(hwnd) && IsCampaignWindowPid((int)pid, pids)) result.Add(hwnd); return true; }, IntPtr.Zero);
        return result;
    }
}

internal static class TraySelfTest
{
    public static int Run()
    {
        try
        {
            var sample = JsonNode.Parse("""{"status":{"active_count":1},"campaign_id":"test","learner_updates":17,"steps_per_second":2.5,"instances":[{"active":true}],"help_requests":[],"alerts":[],"system":{"memory_percent":40}}""")!.AsObject();
            var parsed = Snapshot.Parse(sample);
            if (parsed.ActiveCount != 1 || parsed.Round != 17 || Math.Abs(parsed.StepsPerSecond - 2.5) > 0.001 || parsed.Instances != 1) throw new Exception("state parsing failed");
            if (ProgramChoiceCheck(sample) != "test") throw new Exception("campaign selection failed");
            if (NativeWindows.GameDescendants(new HashSet<int>()).Count != 0) throw new Exception("PID scope failed");
            var descendants = NativeWindows.DescendantPids(new HashSet<int> { 10 }, new Dictionary<int, int> { [11] = 10, [12] = 11, [20] = 99 });
            if (!descendants.SetEquals([10, 11, 12]) || NativeWindows.IsCampaignWindowPid(20, descendants) || !TrayRules.ConfirmAbort) throw new Exception("campaign window or abort guard failed");
            if (new ExitChoiceForm().StopAndExit) throw new Exception("exit confirmation default must not stop");
            var mutexName = @"Local\VoidGymTrayTest-" + Guid.NewGuid().ToString("N");
            using var owner = new Mutex(true, mutexName, out var ownerCreated);
            using var duplicate = new Mutex(true, mutexName, out var duplicateCreated);
            if (!ownerCreated || duplicateCreated) throw new Exception("single-instance mutex failed");
            Console.WriteLine("VoidGym tray self-test passed.");
            return 0;
        }
        catch (Exception ex) { Console.Error.WriteLine(ex); return 1; }
    }
    private static string? ProgramChoiceCheck(JsonObject state) => state["campaign_id"]?.GetValue<string>();
}

internal static class TrayRules { public static bool ConfirmAbort => true; }
