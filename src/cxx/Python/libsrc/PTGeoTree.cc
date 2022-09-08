

#include <VecGeom/base/Transformation3D.h>
#include "PTGeoTree.hh"

using namespace std;

Prompt::GeoTree::GeoTree()
:m_root(std::make_shared<GeoTree::Node>()) {}

Prompt::GeoTree::~GeoTree() {}

shared_ptr<Prompt::GeoTree::Node> Prompt::GeoTree::getRoot() { return m_root; }

std::shared_ptr<Prompt::GeoTree::Node> Prompt::GeoTree::findMotherNodeByPhysical(int num)
{
  return findMotherNodeByPhysical(m_root, num);
}

std::shared_ptr<Prompt::GeoTree::Node> Prompt::GeoTree::findMotherNodeByPhysical(const std::shared_ptr<Prompt::GeoTree::Node> &node, int num)
{
  if(!node)
    return nullptr;
  if(std::find(node->childPhysicalID.begin(), node->childPhysicalID.end(), num) != node->childPhysicalID.end())
    return node;
  for (auto childptr : node->child)
  {
    return findMotherNodeByPhysical(childptr, num);
  }
    return nullptr;
}


shared_ptr<Prompt::GeoTree::Node> Prompt::GeoTree::findNodeByPhysical(int num)
{
  return findNodeByPhysical(m_root, num);
}

shared_ptr<Prompt::GeoTree::Node> Prompt::GeoTree::findNodeByPhysical( const shared_ptr<Prompt::GeoTree::Node> &node, int num)
{
  if(!node)
    return nullptr;
  if (num == node->physical)
    return node;
  for (auto childptr : node->child)
	{
		return findNodeByPhysical(childptr, num);
	}
  	return nullptr;
}

std::vector<std::shared_ptr<Prompt::GeoTree::Node>> Prompt::GeoTree::findNodeByLogical(int num)
{
  std::vector<std::shared_ptr<Node>> logicalnode;
  findNodeByLogical(m_root, num, logicalnode);
  return logicalnode;
}


void Prompt::GeoTree::findNodeByLogical(const std::shared_ptr<Prompt::GeoTree::Node> &node, int num, std::vector<std::shared_ptr<Node>>& logicalnode)
{
  if(!node)
    return;
  if (num == node->logical)
  {
    logicalnode.push_back(node);
  }
  for (auto childptr : node->child)
  {
    findNodeByLogical(childptr, num, logicalnode);
  }
}

void Prompt::GeoTree::printNode(const std::shared_ptr<Prompt::GeoTree::Node> &node, int layer, std::vector<std::vector<int>> &printArray)
{
  try
  {
    printArray.at(layer).push_back(node->physical);
  }
  catch (...)
  {
    printArray.push_back(vector<int>());
    printArray.at(layer).push_back(node->physical);
  }
  for(auto pointptr: node->child)
  {
    printNode(pointptr,layer+1, printArray);
  }
}

void Prompt::GeoTree::print(bool raw)
{
  if(raw)
  {

  }
  else
  {
    vector<vector<int>> printArray;
    printNode(m_root, 0, printArray);
    for(vector<int> varvector : printArray)
    {
      for(int var : varvector)
        cout << var << " ";
      cout << "\n";
    }
  }
}
