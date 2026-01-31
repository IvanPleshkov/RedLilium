# Milestone 1: Foundation

This file describes all requests to claude code related to the milestone.

## Request 1:
Lets start! I want to create a FPS videogame as the result of the project.
We dont create the entire game right now, but we will create the basic structure.
Start with an empty project structure. The lauguage is RUST. Graphics engine will be custom.
Create the workspace with the following crates:
- core: This crate will contain the core logic of the game, including game loop, entity management, and basic utilities.
- graphics: This crate will handle custom rendering engine.
- demos: This crate will contain demo scenes to showcase the capabilities of the engine.
Project contains a single demo with just window creation using winit crate version `0.30.12`.
Demo supports also web target. Keep the web files like html and js in a separate folder inside demos crate.
Provide also a basic README.md file for each crate explaining its purpose. Add to readme the instructions on how configure enviromnent, build and run the demo.
Everything else will be added later.
Imagine also how to handle the documentation for the project. and how to connect the code with the documentation. While creating the documentation, 
think about the best practices to keep the documentation updated with the code changes.
And also keep in mind that claude code can read the documentation and use it to answer questions about the codebase.

## Request 2:
I try to build the web target using instructions from Readme but I get this error. It seems Corgo.toml is missing something. Fix it.
asm-pack build demos --target web --out-dir web/pkg
Error: crate-type must be cdylib to compile to wasm32-unknown-unknown. Add the following to your Cargo.toml file:

## Request 3:
Please change Readme file and provide web run instructions using cargo modules instead of python http server.

## Request 4:
I develop this project using AI agent systems. I need some script to help bots to test the project after each change.
Create a folder scripts/ with a script to do steps:
- Test the build for native target
- Test the build for web target
- Run unit tests in all crates of the workspace
- Run clippy as a linter
Also provide instructions in the Readme file and docs folder on how to use this script.
It's important that the script works in all major operating systems: Linux, Windows and MacOS.
Try to change docs so that claude code will use this script to test the project after each change.

## Request 5:
Add a git hook to check cargo fmt before each commit.
Because docs are used by claude code, skip the formatting check for docs folder.
Add a new file Contributing.md with instructions on how to setup the git hooks and mention that code must be formatted before each commit.

## Request 6:
I need a github actions workflow to test the project on each push and pull request.
There is already a script to test the project, look at Readme and docs for instructions about the script.
The workflow must run on ubuntu-latest, windows-latest and macos-latest.
The workflow should be triggered on each changes to main branch and also on each pull request to any branch.

## Request 7:
Add please a CI step with cargo fmt check to the github actions workflow.

## Request 8:
There is a problem with the github CI workflow.
There is an error in github CI:
Error: Unable to resolve action dtolnay/rust-action, repository not found

## Request 9:
Because the project is a graphics engine, it's important to include graphical tests in the CI workflow.
Please add a CI step with installing a hardware accelerated OpenGL driver on each OS used in the workflow.
And also vulkan emulator.

## Request 10:
Windows github CI cannot install graphics emulators, there is an error:
Extracting archive: C:\Users\RUNNER~1\AppData\Local\Temp\mesa\mesa.7z
Path = C:\Users\RUNNER~1\AppData\Local\Temp\mesa\mesa.7z
Type = 7z
Physical Size = 71852727
Headers Size = 1132
Method = LZMA2:26 LZMA:20 BCJ2
Solid = +
Blocks = 2
Everything is Ok
Files: 49
Size:       656958979
Compressed: 71852727
Copy-Item: D:\a\_temp\0a8fc08d-c4d0-494b-9741-2219789114d2.ps1:15
Line |
  15 |  Copy-Item "$mesaDir\opengl32.dll" -Destination "C:\Windows\System32\" …
     |  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     | Access to the path 'C:\Windows\System32\opengl32.dll' is denied.
Error: Process completed with exit code 1.

## Request 11:
It seems windows github CI is stucked and cannot be finished, here is the full log. Please check is log is correct and fix the problem if it is a problem on our side.
Run # Download Mesa for Windows (software OpenGL via llvmpipe)
    Directory: C:\Users\RUNNER~1\AppData\Local\Temp
Mode                 LastWriteTime         Length Name
d----           1/30/2026 10:00 PM                mesa
7-Zip 25.01 (x64) : Copyright (c) 1999-2025 Igor Pavlov : 2025-08-03
Scanning the drive for archives:
1 file, 71852727 bytes (69 MiB)
Extracting archive: C:\Users\RUNNER~1\AppData\Local\Temp\mesa\mesa.7z
Path = C:\Users\RUNNER~1\AppData\Local\Temp\mesa\mesa.7z
Type = 7z
Physical Size = 71852727
Headers Size = 1132
Method = LZMA2:26 LZMA:20 BCJ2
Solid = +
Blocks = 2
Everything is Ok
Files: 49
Size:       656958979
Compressed: 71852727
    Directory: D:\a\vibeengine\vibeengine
Mode                 LastWriteTime         Length Name
d----           1/30/2026 10:00 PM                mesa-dll

## Request 12:
I see the error in the windows github CI log. Please fix the problem.
Run # Download Mesa for Windows (software OpenGL via llvmpipe)
    Directory: C:\Users\RUNNER~1\AppData\Local\Temp
Mode                 LastWriteTime         Length Name
d----           1/30/2026 10:19 PM                mesa
Scanning the drive for archives:
1 file, 71852727 bytes (69 MiB)
Extracting archive: C:\Users\RUNNER~1\AppData\Local\Temp\mesa\mesa.7z
Path = C:\Users\RUNNER~1\AppData\Local\Temp\mesa\mesa.7z
Type = 7z
Physical Size = 71852727
Headers Size = 1132
Method = LZMA2:26 LZMA:20 BCJ2
Solid = +
Blocks = 2
Everything is Ok
Files: 49
Size:       656958979
Compressed: 71852727
    Directory: D:\a\vibeengine\vibeengine
Mode                 LastWriteTime         Length Name
d----           1/30/2026 10:19 PM                mesa-dll
CodeBase            : file:///C:/Program Files/PowerShell/7/WindowsBase.dll
FullName            : WindowsBase, Version=8.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35
EntryPoint          : 
DefinedTypes        : {Interop, FxResources.WindowsBase.SR, MS.Win32.ExternDll, MS.Win32.HandleCollector…}
IsCollectible       : False
ManifestModule      : WindowsBase.dll
ReflectionOnly      : False
Location            : C:\Program Files\PowerShell\7\WindowsBase.dll
ImageRuntimeVersion : v4.0.30319
GlobalAssemblyCache : False
HostContext         : 0
IsDynamic           : False
ExportedTypes       : {System.Security.RightsManagement.ContentGrant, 
                      System.Security.RightsManagement.SecureEnvironment, 
                      System.Security.RightsManagement.CryptoProvider, 
                      System.Security.RightsManagement.UnsignedPublishLicense…}
IsFullyTrusted      : True
CustomAttributes    : {[System.Runtime.CompilerServices.ExtensionAttribute()], 
                      [System.Runtime.CompilerServices.CompilationRelaxationsAttribute((Int32)8)], 
                      [System.Runtime.CompilerServices.RuntimeCompatibilityAttribute(WrapNonExceptionThrows = True)], [
                      System.Diagnostics.DebuggableAttribute((System.Diagnostics.DebuggableAttribute+DebuggingModes)2)]
                      …}
EscapedCodeBase     : file:///C:/Program%20Files/PowerShell/7/WindowsBase.dll
Modules             : {WindowsBase.dll}
SecurityRuleSet     : None
OperationStopped: 
Error: Process completed with exit code 1.
