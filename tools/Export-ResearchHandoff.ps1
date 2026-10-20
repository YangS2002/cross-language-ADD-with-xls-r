param([string]$Repo = (Split-Path -Parent $PSScriptRoot), [switch]$CleanupLocal, [switch]$Refresh)
$ErrorActionPreference = 'Stop'
$repoPath = (Resolve-Path -LiteralPath $Repo).Path
$outputPath = Join-Path $repoPath 'experiments/handoff_2026-10-11'
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$reports = @(Get-ChildItem -LiteralPath (Join-Path $repoPath 'experiments') -File -Filter '*.md' | Sort-Object Name)
$book = [System.Text.StringBuilder]::new()
[void]$book.AppendLine('# 全部实验报告与研究方案原文汇编（截至2026-10-11）')
[void]$book.AppendLine('本文件保留历史原文，包含被后续要求覆盖的旧规则、计划与阶段性判断。当前有效约束和最终状态以RESEARCH_HANDOFF_2026-10-11.md为准；旧记录中的“权重保留”须结合本次清理清单理解。')
foreach ($file in $reports) {
    [void]$book.AppendLine("`n---`n`n## 原文件：$($file.Name)`n")
    [void]$book.AppendLine([System.IO.File]::ReadAllText($file.FullName))
}
$bookPath = Join-Path $outputPath 'ALL_EXPERIMENT_REPORTS_2026-10-11.md'
[System.IO.File]::WriteAllText($bookPath, $book.ToString(), [System.Text.UTF8Encoding]::new($false))

# Consolidate original evaluation CSVs without changing scores or dropping incomplete domains.
$evaluationRows = [System.Collections.Generic.List[object]]::new()
foreach ($file in (Get-ChildItem -LiteralPath (Join-Path $repoPath 'models') -Recurse -File -Filter 'evaluation_results.csv')) {
    foreach ($row in (Import-Csv -LiteralPath $file.FullName)) {
        $record = [ordered]@{source_file=[System.IO.Path]::GetRelativePath($repoPath,$file.FullName)}
        foreach ($field in 'arm','seed','milestone','checkpoint_epoch','domain','head','eer','threshold','samples','seconds') {
            $record[$field] = $row.$field
        }
        $evaluationRows.Add([PSCustomObject]$record)
    }
}
$evaluationRows | Export-Csv -LiteralPath (Join-Path $outputPath 'ALL_STAGE_EVALUATIONS.csv') -NoTypeInformation -Encoding utf8

$selected = [System.Collections.Generic.List[System.IO.FileInfo]]::new()
foreach ($folder in 'experiments','models','tools','src') {
    foreach ($file in (Get-ChildItem -LiteralPath (Join-Path $repoPath $folder) -Recurse -File)) {
        if ($file.FullName.StartsWith($outputPath + [System.IO.Path]::DirectorySeparatorChar)) { continue }
        $rel = [System.IO.Path]::GetRelativePath($repoPath,$file.FullName)
        if ($rel -match '(^|[\/])(pretrained|__pycache__)([\/]|$)') { continue }
        if ($file.Attributes -band [System.IO.FileAttributes]::ReparsePoint) { throw "Symlink archive member: $rel" }
        if ($file.Extension -notin '.md','.json','.csv','.log','.yaml','.yml','.py','.ps1','.txt','.svg','.html') { continue }
        if ($file.Name -like 'scores*' -or $file.Name -in 'predictions.json','source_energy_samples.csv') { continue }
        $selected.Add($file)
    }
}
foreach ($name in 'train.py','data.py','requirements.txt','.gitignore') {
    $filePath = Join-Path $repoPath $name
    if (Test-Path -LiteralPath $filePath -PathType Leaf) { $selected.Add((Get-Item -LiteralPath $filePath)) }
}
$selected.Add((Get-Item -LiteralPath $bookPath))
$selected.Add((Get-Item -LiteralPath (Join-Path $outputPath 'ALL_STAGE_EVALUATIONS.csv')))
foreach ($name in 'remote_cleanup_plan.json','remote_cleanup_result.json','LOCAL_CLEANUP_RESULT.json') {
    $path = Join-Path $outputPath $name
    if (Test-Path -LiteralPath $path -PathType Leaf) { $selected.Add((Get-Item -LiteralPath $path)) }
}
$archivePath = Join-Path $outputPath 'local_research_snapshot.zip'
if (Test-Path -LiteralPath $archivePath) {
    if (-not $Refresh -or $CleanupLocal) { throw 'Archive already exists; use explicit Refresh without cleanup' }
    $priorManifest = Get-Content -LiteralPath (Join-Path $outputPath 'LOCAL_ARCHIVE_MANIFEST.json') -Raw | ConvertFrom-Json
    if ((Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $priorManifest.archive_sha256) { throw 'Existing generated archive signature changed' }
    $resolvedArchive = (Resolve-Path -LiteralPath $archivePath).Path
    if (-not $resolvedArchive.StartsWith($repoPath + [System.IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase) -or (Get-Item -LiteralPath $resolvedArchive).Attributes -band [System.IO.FileAttributes]::ReparsePoint) { throw 'Unsafe generated archive path' }
    Remove-Item -LiteralPath $resolvedArchive
}
Add-Type -AssemblyName System.IO.Compression
$manifest = [System.Collections.Generic.List[object]]::new()
$archive = [System.IO.Compression.ZipFile]::Open($archivePath, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($file in ($selected | Sort-Object FullName -Unique)) {
        $rel = [System.IO.Path]::GetRelativePath($repoPath,$file.FullName).Replace('\','/')
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive,$file.FullName,$rel,[System.IO.Compression.CompressionLevel]::Optimal) | Out-Null
        $manifest.Add([PSCustomObject]@{path=$rel;bytes=$file.Length;sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()})
    }
} finally { $archive.Dispose() }
$verifyArchive = [System.IO.Compression.ZipFile]::OpenRead($archivePath)
try {
    foreach ($item in $manifest) {
        $entry = $verifyArchive.GetEntry($item.path)
        if ($null -eq $entry -or $entry.Length -ne $item.bytes) { throw "Bad archive member: $($item.path)" }
        $stream = $entry.Open()
        $hasher = [System.Security.Cryptography.SHA256]::Create()
        try { $actual = [Convert]::ToHexString($hasher.ComputeHash($stream)).ToLowerInvariant() }
        finally { $stream.Dispose(); $hasher.Dispose() }
        if ($actual -ne $item.sha256) { throw "Archive member SHA mismatch: $($item.path)" }
    }
} finally { $verifyArchive.Dispose() }
$record = [ordered]@{created_at=[DateTimeOffset]::Now.ToString('o');report_count=$reports.Count;archived_files=$manifest.Count;evaluation_rows=$evaluationRows.Count;archive='local_research_snapshot.zip';archive_bytes=(Get-Item -LiteralPath $archivePath).Length;archive_sha256=(Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant();all_entries_sha256_verified=$true;files=$manifest}
$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $outputPath 'LOCAL_ARCHIVE_MANIFEST.json') -Encoding utf8
[PSCustomObject]@{reports=$reports.Count;archived_files=$manifest.Count;evaluation_rows=$evaluationRows.Count;archive_bytes=$record.archive_bytes;verified=$true} | ConvertTo-Json

if ($CleanupLocal) {
    $resultPath = Join-Path $outputPath 'LOCAL_CLEANUP_RESULT.json'
    if (Test-Path -LiteralPath $resultPath) { throw 'Local cleanup already recorded; no implicit retry' }
    $modelsRoot = (Resolve-Path -LiteralPath (Join-Path $repoPath 'models')).Path
    $explicitFiles = @(
        'M1a_local_transition_preservation_seed20_10ep/checkpoints/seed_20_best_train_loss.pth',
        'M2_dual_transition_second_order_residual_seed20_10ep/checkpoints/seed_20_best_train_loss.pth',
        'M2_dual_transition_second_order_residual_seed1001_10ep/checkpoints/seed_1001_best_train_loss.pth'
    )
    $candidates = @($explicitFiles | ForEach-Object {
        $path = (Resolve-Path -LiteralPath (Join-Path $modelsRoot $_)).Path
        if (-not $path.StartsWith($modelsRoot + [System.IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) { throw "Outside models root: $path" }
        $item = Get-Item -LiteralPath $path
        if ($item.PSIsContainer -or $item.Extension -ne '.pth') { throw "Invalid cleanup file: $path" }
        $ancestor = $item
        while ($null -ne $ancestor) {
            if ($ancestor.Attributes -band [System.IO.FileAttributes]::ReparsePoint) { throw "Reparse point in cleanup path: $path" }
            $ancestor = if ($ancestor -is [System.IO.FileInfo]) { $ancestor.Directory } else { $ancestor.Parent }
        }
        [PSCustomObject]@{path=$path;bytes=$item.Length;last_write_ticks=$item.LastWriteTimeUtc.Ticks;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
    })
    $result = [ordered]@{status='deleting';started_at=[DateTimeOffset]::Now.ToString('o');archive_verified=$true;candidates=$candidates;deleted=@();deleted_bytes=0;recovery='Deleted training binaries were not archived and cannot be recovered from this records bundle'}
    $result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $resultPath -Encoding utf8
    try {
        foreach ($item in $candidates) {
            $current = Get-Item -LiteralPath $item.path
            if ($current.Length -ne $item.bytes -or $current.LastWriteTimeUtc.Ticks -ne $item.last_write_ticks) { throw 'Local candidate changed since validation' }
            Remove-Item -LiteralPath $item.path
            $result.deleted += $item
            $result.deleted_bytes += $item.bytes
            $result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $resultPath -Encoding utf8
        }
        $result.status = 'complete'
        $result.ended_at = [DateTimeOffset]::Now.ToString('o')
        $result.remaining = @($candidates | Where-Object { Test-Path -LiteralPath $_.path })
    } catch {
        $result.status = 'partial_failure'
        $result.error = $_.Exception.Message
        throw
    } finally {
        $result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $resultPath -Encoding utf8
    }
    [PSCustomObject]@{local_cleanup_status=$result.status;deleted_files=$result.deleted.Count;deleted_bytes=$result.deleted_bytes;remaining=$result.remaining.Count} | ConvertTo-Json
}
