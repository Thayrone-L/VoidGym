using Mono.Cecil;

if (args.Length < 2)
{
    Console.Error.WriteLine("Usage: HkAssemblyInspector <assembly> <type-pattern> [...]");
    return 2;
}

string assemblyPath = Path.GetFullPath(args[0]);
string[] patterns = args.Skip(1).ToArray();
using AssemblyDefinition assembly = AssemblyDefinition.ReadAssembly(assemblyPath);

foreach (TypeDefinition type in AllTypes(assembly.MainModule.Types)
             .Where(type => patterns.Any(pattern => Matches(type.FullName, pattern))))
{
    Console.WriteLine($"TYPE {type.FullName} : {type.BaseType?.FullName ?? "<none>"}");
    foreach (FieldDefinition field in type.Fields)
    {
        Console.WriteLine($"  FIELD {FieldVisibility(field)} {field.FieldType.FullName} {field.Name}");
    }
    foreach (PropertyDefinition property in type.Properties)
    {
        Console.WriteLine($"  PROPERTY {property.PropertyType.FullName} {property.Name}");
    }
    foreach (MethodDefinition method in type.Methods)
    {
        string parameters = string.Join(", ", method.Parameters.Select(parameter =>
            $"{parameter.ParameterType.FullName} {parameter.Name}"));
        Console.WriteLine(
            $"  METHOD {MethodVisibility(method)} {method.ReturnType.FullName} {method.Name}({parameters})"
        );
    }
}

return 0;

static IEnumerable<TypeDefinition> AllTypes(IEnumerable<TypeDefinition> roots)
{
    foreach (TypeDefinition type in roots)
    {
        yield return type;
        foreach (TypeDefinition nested in AllTypes(type.NestedTypes))
        {
            yield return nested;
        }
    }
}

static bool Matches(string typeName, string pattern) =>
    pattern.StartsWith("=", StringComparison.Ordinal)
        ? string.Equals(typeName, pattern[1..], StringComparison.OrdinalIgnoreCase)
        : typeName.Contains(pattern, StringComparison.OrdinalIgnoreCase);

static string FieldVisibility(FieldDefinition field) => field.IsPublic ? "public" : "nonpublic";

static string MethodVisibility(MethodDefinition method) => method.IsPublic ? "public" : "nonpublic";
